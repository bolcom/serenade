//! Shared path rules for configuration files and command-line overrides.
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

pub const DEFAULT_INDEX_CAPACITY: usize = 2500;

pub fn absolute(path: impl AsRef<Path>) -> Result<PathBuf> {
    let path = path.as_ref();
    Ok(if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()?.join(path)
    })
}
pub fn relative_to(config: &Path, path: impl AsRef<Path>) -> Result<PathBuf> {
    let path = path.as_ref();
    absolute(if path.is_absolute() {
        path.to_owned()
    } else {
        config.parent().unwrap_or(Path::new(".")).join(path)
    })
}
pub fn input(path: impl AsRef<Path>) -> Result<PathBuf> {
    let path = absolute(path)?;
    anyhow::ensure!(
        path.is_file(),
        "required input file does not exist: {}",
        path.display()
    );
    std::fs::canonicalize(&path).with_context(|| format!("resolve input {}", path.display()))
}

/// Training may also be a legacy persisted index directory.
pub fn corpus(path: impl AsRef<Path>) -> Result<PathBuf> {
    let path = absolute(path)?;
    anyhow::ensure!(
        path.is_file() || path.is_dir(),
        "required training data or index does not exist: {}",
        path.display()
    );
    std::fs::canonicalize(&path)
        .with_context(|| format!("resolve training input {}", path.display()))
}
