//! Slash command definitions. Discord identity and roles are forwarded to the engine, which
//! builds `RequestContext`.

use serenity::all::{CommandOptionType, CreateCommand, CreateCommandOption};

/// `/ask <question>`.
pub const ASK: &str = "ask";
/// `/reset` — start a fresh conversation.
pub const RESET: &str = "reset";
/// Name of the question option on `/ask`.
pub const QUESTION: &str = "question";

/// Every command the bot registers on its guild.
pub fn all() -> Vec<CreateCommand> {
    vec![
        CreateCommand::new(ASK)
            .description("Ask Sparky about ASU: hours, events, clubs, courses, and more")
            .add_option(
                CreateCommandOption::new(CommandOptionType::String, QUESTION, "Your question")
                    .required(true),
            ),
        CreateCommand::new(RESET).description("Forget this conversation and start over"),
    ]
}
