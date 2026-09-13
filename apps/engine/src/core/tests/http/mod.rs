//! The HTTP surface: the OpenAI-compatible API and rate limiting.

mod openai;
mod rate_limit;

#[test]
fn a_secret_matches_only_when_every_byte_does() {
    use crate::routes::chat::same_secret;

    assert!(same_secret(b"change-me", b"change-me"));
    assert!(!same_secret(b"change-mf", b"change-me"));
    assert!(!same_secret(b"change", b"change-me"));
    assert!(!same_secret(b"", b"change-me"));
}
