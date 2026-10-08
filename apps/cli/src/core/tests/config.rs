//! Console settings and the repo root.

use crate::core::config::{Cli, repo_root};

#[test]
fn cli_defaults_point_at_local_phoenix() {
    assert_eq!(Cli::default().phoenix_url, "http://localhost:6006");
}

#[test]
fn repo_root_is_found_from_a_nested_directory() {
    let here = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = repo_root(here).unwrap_or_default();
    assert!(root.join("justfile").is_file());
    assert!(repo_root(std::path::Path::new("/")).is_none());
}
