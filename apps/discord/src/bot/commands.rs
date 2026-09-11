//! Slash command definitions. Discord identity and roles are forwarded to the engine, which
//! builds RequestContext.

use serenity::all::{CommandOptionType, CreateCommand, CreateCommandOption};

/// The /ask command, which takes a question.
pub const ASK: &str = "ask";
/// The /reset command, which ends the conversation in this channel.
pub const RESET: &str = "reset";
/// The /memory command, which lists what the engine remembers about the user.
pub const MEMORY: &str = "memory";
/// The /forget command, which removes one remembered thing or everything.
pub const FORGET: &str = "forget";
/// Name of the question option on /ask.
pub const QUESTION: &str = "question";
/// Name of the private option on /ask.
pub const PRIVATE: &str = "private";
/// Name of the label option on /forget.
pub const LABEL: &str = "label";

/// Every command the bot registers on its guild.
pub fn all() -> Vec<CreateCommand> {
    vec![
        CreateCommand::new(ASK)
            .description("Ask Sparky about ASU: hours, events, clubs, courses, and more")
            .add_option(
                CreateCommandOption::new(CommandOptionType::String, QUESTION, "Your question")
                    .required(true),
            )
            .add_option(CreateCommandOption::new(
                CommandOptionType::Boolean,
                PRIVATE,
                "Only you see the answer, and Sparky may use what it remembers about you",
            )),
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
    ]
}
