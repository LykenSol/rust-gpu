#[cfg(test)]
mod test;

mod depfile;
mod rustc;
mod source;
mod utils;

use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::{env, fmt};

use eyre::Result;

pub use source::{Source, SourceKind};

#[derive(Debug)]
pub enum SpirvBuilderError {
    BuildFailed,
}

impl fmt::Display for SpirvBuilderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpirvBuilderError::BuildFailed => f.write_str("Build failed"),
        }
    }
}

impl Error for SpirvBuilderError {}

pub enum MemoryModel {
    Simple,
    Vulkan,
    GLSL450,
}

impl fmt::Display for MemoryModel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                MemoryModel::Simple => "simple",
                MemoryModel::Vulkan => "vulkan",
                MemoryModel::GLSL450 => "glsl450",
            }
        )
    }
}

#[derive(Clone, Copy)]
pub struct Version {
    major: u8,
    minor: u8,
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "spirv{}.{}", self.major, self.minor)
    }
}

#[derive(Default)]
pub struct SpirvBuilder {
    build_script_metadata: bool,
    codegen_options: CodegenBuildOptions,
    codegen_source: Source,
    destination: PathBuf,
    memory_model: Option<MemoryModel>,
    rust_src: Option<PathBuf>,
    sysroot_location: Source,
    source: PathBuf,
    spirv_version: Option<Version>,
}

impl SpirvBuilder {
    pub fn new(source: impl Into<PathBuf>, destination: impl Into<PathBuf>) -> Self {
        Self {
            source: source.into(),
            destination: destination.into(),
            ..Default::default()
        }
    }

    /// Sets the source for code generation.
    pub fn codegen_source(mut self, codegen_source: Source) -> Self {
        self.codegen_source = codegen_source;
        self
    }

    /// Sets the location for the sysroot.
    pub fn sysroot_location(mut self, sysroot_location: Source) -> Self {
        self.sysroot_location = sysroot_location;
        self
    }

    /// Sets the SPIR-V binary version to use. Defaults to v1.3.
    pub fn spirv_version(mut self, major: u8, minor: u8) -> Self {
        self.spirv_version = Some(Version { major, minor });
        self
    }

    /// Sets the path to the rust source to use for building sysroot. Default:
    /// The current toolchain's `rust-src` component.
    pub fn rust_src(mut self, rust_src: impl Into<PathBuf>) -> Self {
        self.rust_src = Some(rust_src.into());
        self
    }

    /// Sets the SPIR-V memory model. Defaults to Vulkan.
    pub fn memory_model(mut self, memory_model: MemoryModel) -> Self {
        self.memory_model = Some(memory_model);
        self
    }

    /// Sets whether to print build script metadata (e.g. `rerun-if-changed`
    /// for artifacts) to `cargo`.
    pub fn emit_build_script_metadata(mut self) -> Self {
        self.build_script_metadata = true;
        self
    }

    /// Creates the target feature flag for rustc, setting the SPIR-V version
    /// and/or memory model.
    fn target_feature_flag(&self) -> String {
        let mut target_features = Vec::new();
        if let Some(version) = self.spirv_version {
            target_features.push(format!("+{}", version));
        }
        if let Some(memory_model) = &self.memory_model {
            target_features.push(format!("+{}", memory_model));
        }

        if target_features.is_empty() {
            String::new()
        } else {
            format!("-C target-feature={}", target_features.join(","))
        }
    }

    fn rustc_flags(&self, sysroot: Option<impl AsRef<Path>>, codegen: impl AsRef<Path>) -> String {
        vec![
            match sysroot {
                Some(path) => format!("--sysroot={}", path.as_ref().display()),
                None => String::new(),
            },
            format!("-Zcodegen-backend={}", codegen.as_ref().display()),
            self.target_feature_flag(),
        ]
        .join(" ")
    }

    /// Builds the module. Returns the path to the built spir-v file(s).
    pub fn build(self) -> Result<PathBuf> {
        if std::fs::metadata(&self.destination).is_ok() {
            remove_dir_all::remove_dir_all(&self.destination)?;
        }

        std::fs::create_dir_all(&self.destination)?;

        for module in self.build_shader()? {
            std::fs::copy(&module, self.destination.join(module.file_name().unwrap()))?;
        }

        Ok(self.destination)
    }

    /// Builds the SPIR-V codegen backend and returns a path to the
    /// compiled dynamic library. If `codegen_source.compile_source` is `false`
    /// then then it will return the `codegen_source` path with the DLL file name.
    pub fn build_spirv_codegen(&self) -> Result<PathBuf> {
        const CRATE_NAME: &str = "rustc_codegen_spirv";
        let dll_file_name = format!(
            "{}{}{}",
            env::consts::DLL_PREFIX,
            CRATE_NAME,
            env::consts::DLL_SUFFIX
        );

        let repo_dir = match self.codegen_source.get() {
            Some(path) if self.codegen_source.compile_source => path,
            Some(path) => return Ok(path.join(&dll_file_name)),
            None => return Ok(PathBuf::from("spirv")),
        };

        let mut args = vec![
            "build".into(),
            format!("--features={}", self.codegen_options.spirv_tools),
        ];

        if self.codegen_options.release {
            args.push("--release".into());
        }

        let output = rustc::cargo()
            .args(&args)
            .stderr(Stdio::inherit())
            .current_dir(&repo_dir.join("crates").join(CRATE_NAME))
            .output()?;

        let backend = repo_dir
            .join("target")
            .join(if self.codegen_options.release {
                "release"
            } else {
                "debug"
            })
            .join(&dll_file_name);

        if !output.status.success() || std::fs::metadata(&backend).is_err() {
            Err(eyre::eyre!("Building `rustc_codegen_spirv` failed."))
        } else {
            Ok(backend)
        }
    }

    pub fn build_sysroot(&self, backend: impl AsRef<Path>) -> Result<Option<PathBuf>> {
        const RUSTFLAGS: &str = "RUSTFLAGS";
        const TARGET_NAME: &str = "spirv-unknown-unknown";
        let backend = backend.as_ref();
        let flags = format!("-Zcodegen-backend={}", backend.display());
        let src = match &self.rust_src {
            Some(path) => path.clone(),
            None => utils::get_rust_src()?,
        };

        let target_dir = match self.sysroot_location.get() {
            Some(path) if self.sysroot_location.compile_source => path,
            Some(path) => return Ok(Some(path)),
            None => return Ok(None),
        };

        let previous_flags = env::var(RUSTFLAGS);
        env::set_var(RUSTFLAGS, flags);
        let sysroot = cargo_sysroot::build_sysroot_with(
            None,
            &target_dir,
            TARGET_NAME.as_ref(),
            &src,
            cargo_sysroot::Sysroot::CompilerBuiltins,
            false,
        )
        .map_err(|e| eyre::eyre!("{}", e))?;
        env::remove_var(RUSTFLAGS);
        if let Ok(flags) = previous_flags {
            env::set_var(RUSTFLAGS, flags);
        }

        let sysroot_codegen = sysroot.join(backend.file_name().unwrap());
        std::fs::copy(&backend, &sysroot_codegen)?;
        Ok(Some(sysroot))
    }

    pub fn build_shader(&self) -> eyre::Result<Vec<PathBuf>> {
        let backend = self.build_spirv_codegen()?;
        let sysroot = self.build_sysroot(&backend)?;
        let flags = self.rustc_flags(sysroot, backend);
        let build = rustc::cargo()
            .args(&[
                "build",
                "--verbose",
                "--release",
                "--message-format=json-render-diagnostics",
                "--target",
                "spirv-unknown-unknown",
            ])
            .stderr(Stdio::inherit())
            .current_dir(&self.source)
            .env("RUSTFLAGS", flags)
            .output()
            .expect("failed to execute cargo build");

        if build.status.success() {
            let stdout = String::from_utf8(build.stdout).unwrap();
            let artifacts = rustc::get_last_artifact(&stdout);
            if self.build_script_metadata {
                for artifact in &artifacts {
                    rustc::print_deps_of(&artifact);
                }
            }

            Ok(artifacts)
        } else {
            Err(SpirvBuilderError::BuildFailed.into())
        }
    }
}

#[derive(Default)]
pub struct CodegenBuildOptions {
    pub release: bool,
    pub spirv_tools: SpirvToolsFeatures,
}

pub enum SpirvToolsFeatures {
    Auto,
    Compiled,
    Installed,
}

impl Default for SpirvToolsFeatures {
    fn default() -> Self {
        Self::Auto
    }
}

impl fmt::Display for SpirvToolsFeatures {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        const COMPILED_FEATURE: &str = "use-compiled-tools";
        const INSTALLED_FEATURE: &str = "use-installed-tools";
        write!(
            f,
            "{}",
            match self {
                Self::Auto => {
                    if std::process::Command::new("spirv-val")
                        .args(&["-h"])
                        .status()
                        .map_or(false, |s| s.success())
                    {
                        INSTALLED_FEATURE
                    } else {
                        COMPILED_FEATURE
                    }
                }
                Self::Compiled => COMPILED_FEATURE,
                Self::Installed => INSTALLED_FEATURE,
            }
        )
    }
}
