use crate::core::config::Agent;

#[test]
fn traces_default_to_the_sparky_state_directory() {
    assert_eq!(Agent::default().trace_dir, ".sparky/traces");
}
