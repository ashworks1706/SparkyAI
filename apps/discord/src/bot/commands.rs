//! Slash commands for memory and conversation state.

use serenity::all::{CommandOptionType, CreateCommand, CreateCommandOption};

/// The /reset command, which ends the conversation in this channel.
pub const RESET: &str = "reset";
/// The /memory command, which lists what the engine remembers about the user.
pub const MEMORY: &str = "memory";
/// The /forget command, which removes one remembered thing or everything.
pub const FORGET: &str = "forget";
/// Name of the label option on /forget.
pub const LABEL: &str = "label";
/// The /login command, which connects one of the user's accounts.
pub const LOGIN: &str = "login";
/// The /logout command, which disconnects one of the user's accounts.
pub const LOGOUT: &str = "logout";
/// Name of the service option on /login and /logout.
pub const SERVICE: &str = "service";
/// The services a login connects: the option value is the provider key the engine expects.
pub const SERVICES: [(&str, &str); 2] = [("Canvas", "canvas"), ("Outlook", "microsoft")];

/// Every command the bot registers on its guild.
pub fn all() -> Vec<CreateCommand> {
    vec![
        CreateCommand::new(RESET)
            .description("End the conversation in this channel and start over"),
        CreateCommand::new(MEMORY).description("See what Sparky remembers about you"),
        CreateCommand::new(FORGET)
            .description("Make Sparky forget one thing, or everything, about you")
            .add_option(CreateCommandOption::new(
                CommandOptionType::String,
                LABEL,
                "The thing to forget, as /memory shows it. Leave empty to forget everything",
            )),
        CreateCommand::new(LOGIN)
            .description("Connect an account (Canvas, Outlook) so Sparky can check it in DMs")
            .add_option(service_option()),
        CreateCommand::new(LOGOUT)
            .description("Disconnect an account (Canvas, Outlook) from Sparky")
            .add_option(service_option()),
    ]
}

/// The required service choice shared by /login and /logout.
fn service_option() -> CreateCommandOption {
    let mut option = CreateCommandOption::new(
        CommandOptionType::String,
        SERVICE,
        "Which account to connect or disconnect",
    )
    .required(true);
    for (label, value) in SERVICES {
        option = option.add_string_choice(label, value);
    }
    option
}
