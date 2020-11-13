use std::path::PathBuf;

use eyre::Result;

fn get_rustc_sysroot() -> Result<PathBuf> {
    let output = crate::rustc::rustc()
        .args(&["--print", "sysroot"])
        .output()?;
    Ok(PathBuf::from(std::str::from_utf8(&output.stdout)?.trim()))
}

/// Get the rust-src stuff
pub fn get_rust_src() -> Result<PathBuf> {
    // See <https://github.com/rust-lang/rustup#can-rustup-download-the-rust-source-code>
    Ok(get_rustc_sysroot()?
        .join("lib")
        .join("rustlib")
        .join("src")
        .join("rust")
        .join("library"))
}
