use std::path::PathBuf;

/// Defines the source to use for `rustc_codegen_spirv` and `sysroot`.
#[derive(Debug, Default)]
pub struct Source {
    /// Whether to attempt to compile the source provided by `kind`. If `false`
    /// expects the source to point to a prebuilt sysroot containing the
    /// `spirv-unknown-unknown` sysroot and codegen. Errors if `true`, and
    /// using `SourceKind::Environment`.
    pub compile_source: bool,
    pub kind: SourceKind,
}

impl Source {
    pub fn get(&self) -> Option<PathBuf> {
        match &self.kind {
            SourceKind::Environment => None,
            SourceKind::Path(path) => Some(path.clone()),
            SourceKind::Git {
                repository,
                branch,
                into,
            } => {
                git2::build::RepoBuilder::new()
                    .branch(&branch)
                    .clone(&repository, &into)
                    .ok()?;
                Some(into.clone())
            }
        }
    }
}

/// Where the source is located.
#[derive(Debug)]
pub enum SourceKind {
    /// Clones a git repository for the source.
    Git {
        /// The git repository.
        repository: String,
        /// The branch to checkout (Default: `main`).
        branch: String,
        /// The directory to clone the repository into.
        into: PathBuf,
    },
    /// Uses the path provided for the source.
    Path(PathBuf),
    /// Assume that `rustc_codegen_spirv` and `spirv-unknown-unknown` are
    /// already available in the environment.
    Environment,
}

impl Default for SourceKind {
    fn default() -> Self {
        Self::Environment
    }
}
