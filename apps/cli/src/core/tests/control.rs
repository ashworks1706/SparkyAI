//! Command line parsing.

use crate::app::control::parse_command;
use crate::core::types::Command;

#[test]
fn commands_parse_into_actions() {
    assert_eq!(parse_command("q"), Command::Quit);
    assert_eq!(
        parse_command("start engine"),
        Command::Start("engine".into())
    );
    assert_eq!(parse_command("stop  chat"), Command::Stop("chat".into()));
    assert_eq!(
        parse_command("restart discord"),
        Command::Restart("discord".into())
    );
    assert_eq!(parse_command("help"), Command::Help);
}

#[test]
fn unrecognised_words_become_just_recipes() {
    assert_eq!(
        parse_command("eval run"),
        Command::Just(vec!["eval".into(), "run".into()])
    );
    assert_eq!(
        parse_command("just check"),
        Command::Just(vec!["check".into()])
    );
    assert_eq!(parse_command("start"), Command::Just(vec!["start".into()]));
    assert_eq!(parse_command("   "), Command::Unknown(String::new()));
}
