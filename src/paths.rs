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
        display(&path)
    );
    std::fs::canonicalize(&path).with_context(|| format!("resolve input {}", display(&path)))
}

/// Training may also be a legacy persisted index directory.
pub fn corpus(path: impl AsRef<Path>) -> Result<PathBuf> {
    let path = absolute(path)?;
    anyhow::ensure!(
        path.is_file() || path.is_dir(),
        "required training data or index does not exist: {}",
        display(&path)
    );
    std::fs::canonicalize(&path)
        .with_context(|| format!("resolve training input {}", display(&path)))
}

/// Human-readable paths for messages only; never use this to open a file.
pub fn display(path: impl AsRef<Path>) -> String {
    let text = path.as_ref().to_string_lossy().into_owned();
    #[cfg(windows)]
    {
        if let Some(rest) = text.strip_prefix(r"\\?\UNC\") {
            return format!(r"\\{}", rest);
        }
        if let Some(rest) = text.strip_prefix(r"\\?\") {
            // Only conventional drive paths have an equivalent ordinary spelling.
            if rest.as_bytes().get(1) == Some(&b':') {
                return rest.to_owned();
            }
        }
    }
    text
}

#[cfg(test)]
mod display_tests {
    #[test]
    fn ordinary_paths_are_unchanged() {
        assert_eq!(super::display("data/train.txt"), "data/train.txt");
    }
    #[cfg(windows)]
    #[test]
    fn windows_verbatim_paths_are_readable() {
        assert_eq!(
            super::display(r"\\?\C:\data\train.txt"),
            r"C:\data\train.txt"
        );
        assert_eq!(
            super::display(r"\\?\UNC\server\share\train.txt"),
            r"\\server\share\train.txt"
        );
        assert_eq!(
            super::display(r"\\?\Volume{abc}\file"),
            r"\\?\Volume{abc}\file"
        );
    }
}
