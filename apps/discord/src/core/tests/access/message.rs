//! What a message carries: mentions, the bot role, quotes, attachments, and thread names.

use serenity::all::{RoleId, UserId};

use crate::access::message::{
    THREAD_NAME_MAX, addresses_bot, bot_role, files, images, quoted, strip_mentions, thread_name,
};

#[test]
fn mentions_of_the_bot_are_stripped_and_others_kept() {
    let me = UserId::new(42);
    assert_eq!(
        strip_mentions("<@42> when does Hayden close?", me, None),
        "when does Hayden close?"
    );
    assert_eq!(
        strip_mentions("hey <@!42> ask <@7>", me, None),
        "hey   ask <@7>"
    );
    assert_eq!(strip_mentions("  <@42> <@!42> ", me, None), "");
}

#[test]
fn mentions_of_the_bot_role_are_stripped_and_other_roles_kept() {
    let me = UserId::new(42);
    let role = Some(RoleId::new(9));
    assert_eq!(strip_mentions("<@&9> hi", me, role), "hi");
    assert_eq!(strip_mentions("<@&8> hi", me, role), "<@&8> hi");
}

#[test]
fn the_bot_role_is_the_one_tagged_with_the_bot() {
    let me = UserId::new(42);
    let roles = [
        (RoleId::new(1), None),
        (RoleId::new(2), Some(UserId::new(7))),
        (RoleId::new(3), Some(me)),
    ];
    let role = bot_role(roles, me);
    assert_eq!(role, Some(RoleId::new(3)));
    assert_eq!(bot_role([(RoleId::new(1), None)], me), None);

    assert!(addresses_bot(false, &[RoleId::new(3)], role));
    assert!(addresses_bot(true, &[], role));
    assert!(!addresses_bot(false, &[RoleId::new(2)], role));
    assert!(!addresses_bot(false, &[RoleId::new(3)], None));
}

#[test]
fn thread_names_fit_discord_and_are_never_empty() {
    assert_eq!(
        thread_name("  when does\n Hayden close? "),
        "when does Hayden close?"
    );
    assert_eq!(thread_name("   "), "Question");
    let long = thread_name(&"é".repeat(300));
    assert_eq!(long.chars().count(), THREAD_NAME_MAX);
    assert!(long.ends_with("..."));
}

#[test]
fn only_a_reply_to_the_bots_own_words_is_quoted_back() {
    let me = UserId::new(9);
    let someone = UserId::new(10);

    assert_eq!(
        quoted(Some(me), " The library closes at 10pm. ", me),
        Some("The library closes at 10pm.".to_owned())
    );
    // A reply to a person is theirs, not a turn for the bot.
    assert_eq!(quoted(Some(someone), "what do you think?", me), None);
    // A reply to nothing, and a reply to a message that carries no text.
    assert_eq!(quoted(None, "text", me), None);
    assert_eq!(quoted(Some(me), "   ", me), None);
}

#[test]
fn only_attachments_discord_calls_an_image_reach_the_model() {
    let sent = images(
        [
            ("https://cdn/one.png", Some("image/png")),
            ("https://cdn/sheet.csv", Some("text/csv")),
            ("https://cdn/two.jpg", Some("image/jpeg; charset=binary")),
            // Discord reports no type for some uploads; a guess is not made for it.
            ("https://cdn/three", None),
        ],
        10,
    );

    assert_eq!(
        sent.iter().map(|a| a.url.as_str()).collect::<Vec<_>>(),
        ["https://cdn/one.png", "https://cdn/two.jpg"]
    );
    // The parameter is dropped, so the media type is the one the model is sent.
    assert_eq!(sent[1].media_type, "image/jpeg");
}

#[test]
fn no_more_than_max_images_of_one_message_are_sent() {
    let many: Vec<(&str, Option<&str>)> = (0..10)
        .map(|_| ("https://cdn/x.png", Some("image/png")))
        .collect();

    assert_eq!(images(many.clone(), 4).len(), 4);
    assert_eq!(images(many, 0).len(), 0);
}

#[test]
fn files_that_are_not_images_are_sent_to_the_engine_within_the_cap() {
    let sent = files(
        [
            (
                "https://cdn/syllabus.pdf",
                "syllabus.pdf",
                Some("application/pdf"),
                120_000,
            ),
            (
                "https://cdn/photo.png",
                "photo.png",
                Some("image/png"),
                5_000,
            ),
            (
                "https://cdn/huge.zip",
                "huge.zip",
                Some("application/zip"),
                9_000_000,
            ),
            ("https://cdn/notes", "notes", None, 300),
        ],
        4,
        2_000_000,
    );
    let names: Vec<&str> = sent.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(
        names,
        ["syllabus.pdf", "notes"],
        "an image goes as an image, and a file over the cap is left out"
    );
    assert_eq!(sent[0].media_type, "application/pdf");
    assert_eq!(
        files([("https://cdn/a.pdf", "a.pdf", None, 1)], 0, 2_000_000).len(),
        0
    );
}
