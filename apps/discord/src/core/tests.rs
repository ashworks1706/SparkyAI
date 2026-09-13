//! Bot unit tests for rendering, routing, roles, component ids, analytics, and span export.

use uuid::Uuid;

use crate::access::roles::can_write;
use crate::core::types::{ChatResponse, Citation};
use crate::render::reply::{MAX_MESSAGE, chunk};
use serenity::all::Permissions;

fn response(text: &str, citations: Vec<Citation>, status: &str) -> ChatResponse {
    ChatResponse {
        request_id: Uuid::new_v4(),
        conversation_id: Uuid::new_v4(),
        text: text.into(),
        citations,
        confirmation: None,
        status: status.into(),
        memories: Vec::new(),
    }
}

/// A source that carries a link, so it renders as a button.
fn linked(title: &str) -> Citation {
    Citation {
        title: title.into(),
        url: Some(format!("https://asu.edu/{title}")),
    }
}

/// A source with no page of its own, so it renders as text.
fn unlinked(title: &str) -> Citation {
    Citation {
        title: title.into(),
        url: None,
    }
}

/// The label of every button in one row.
fn labels_of(row: &[crate::render::components::ButtonSpec]) -> Vec<String> {
    use crate::render::components::ButtonSpec;
    row.iter()
        .map(|b| match b {
            ButtonSpec::Press { label, .. } => (*label).to_owned(),
            ButtonSpec::Link { label, .. } => label.clone(),
        })
        .collect()
}

/// The final render of resp with no head and no steps, as one string.
fn render(resp: &ChatResponse, limit: usize) -> Vec<String> {
    crate::render::card::answer(&[], resp, limit)
}

#[test]
fn short_text_is_one_message() {
    assert_eq!(chunk("hello", MAX_MESSAGE), vec!["hello".to_owned()]);
}

#[test]
fn long_text_splits_on_line_boundaries_under_the_limit() {
    let text = (0..200)
        .map(|i| format!("line {i} {}", "x".repeat(40)))
        .collect::<Vec<_>>()
        .join("\n");
    let parts = chunk(&text, 500);
    assert!(parts.len() > 1);
    assert!(parts.iter().all(|p| p.len() <= 500));
    assert!(parts.iter().all(|p| !p.ends_with('\n')));
    let rejoined = parts.join("\n");
    assert!(rejoined.contains("line 199"));
}

#[test]
fn a_linked_source_becomes_a_button_and_an_unlinked_one_a_subtext_line() {
    use crate::render::components::{ButtonSpec, rows_for};

    let resp = response(
        "2am",
        vec![linked("library_hours"), unlinked("front desk note")],
        "answered",
    );
    let out = render(&resp, 2_000);
    assert_eq!(out.len(), 1);
    assert!(!out[0].contains("Sources"), "no source list in the text");
    assert!(!out[0].contains("fetched"), "no fetch dates either");
    assert!(out[0].contains("Also from** front desk note"), "{}", out[0]);

    let rows = rows_for(&resp);
    assert_eq!(rows.len(), 1, "one row of sources");
    assert_eq!(
        rows[0],
        vec![ButtonSpec::Link {
            label: "library hours".into(),
            url: "https://asu.edu/library_hours".into(),
        }],
        "only the source with a page of its own is a button"
    );
}

#[test]
fn a_button_label_stays_within_what_discord_shows() {
    use crate::render::card::source_label;

    assert_eq!(source_label("library_hours"), "library hours");
    let long = source_label(&"a".repeat(80));
    assert!(long.chars().count() <= 40, "{long}");
    assert!(long.ends_with('\u{2026}'), "{long}");
}

#[test]
fn empty_answer_explains_the_status() {
    let out = render(&response("", vec![], "deadline"), 2_000);
    assert!(out[0].contains("too long"));
}

#[test]
fn no_traceparent_without_an_active_span() {
    use crate::engine::client::current_traceparent;

    assert!(current_traceparent().is_none());
}

#[test]
fn discord_management_permissions_grant_write_access() {
    assert!(can_write(Permissions::MANAGE_GUILD));
    assert!(can_write(Permissions::ADMINISTRATOR));
    assert!(!can_write(Permissions::MANAGE_MESSAGES));
}

#[test]
fn a_guild_role_cannot_impersonate_the_write_capability() {
    use crate::access::roles::authorized_roles;
    use crate::core::config::WRITE_CAPABILITY;

    let named = vec!["students".to_owned(), WRITE_CAPABILITY.to_owned()];
    assert_eq!(
        authorized_roles(
            named.clone(),
            Some(Permissions::MANAGE_MESSAGES),
            WRITE_CAPABILITY
        ),
        vec!["students".to_owned()]
    );
    assert_eq!(
        authorized_roles(named, Some(Permissions::MANAGE_GUILD), WRITE_CAPABILITY),
        vec!["students".to_owned(), WRITE_CAPABILITY.to_owned()]
    );
    assert_eq!(
        authorized_roles(vec!["students".to_owned()], None, WRITE_CAPABILITY),
        vec!["students".to_owned()]
    );
}

#[test]
fn capacity_and_outage_read_differently_to_the_user() {
    use crate::core::types::EngineError;
    use crate::render::reply::failure;

    let busy = failure(&EngineError::Status {
        status: 503,
        body: "the model is at capacity".into(),
    });
    assert!(busy.contains("busy"), "{busy}");

    let down = failure(&EngineError::Transport("connection refused".into()));
    assert!(down.contains("unavailable"), "{down}");

    let broken = failure(&EngineError::Status {
        status: 502,
        body: String::new(),
    });
    assert!(broken.contains("unavailable"), "{broken}");

    let gone = failure(&EngineError::Status {
        status: 404,
        body: "no such conversation".into(),
    });
    assert!(gone.contains("Ask again"), "{gone}");
}

#[test]
fn steps_append_as_subtext_and_an_immediate_repeat_collapses() {
    use crate::render::card::{Steps, THINKING, thinking};

    let mut steps = Steps::default();
    assert!(steps.push(None, "thinking"));
    assert!(
        !steps.push(None, "thinking"),
        "an immediate repeat is dropped"
    );
    assert!(steps.push(None, "searching live courses"));
    assert!(steps.push(None, "thinking"), "a repeat later on is kept");
    assert!(!steps.push(None, "   "));
    assert_eq!(steps.lines().len(), 3);

    let head = thinking(&[], 2_000, 0);
    assert!(head.ends_with(THINKING), "{head}");
    let shown = thinking(&steps.lines(), 2_000, 0);
    assert_eq!(
        shown,
        format!("{head}\n-# thinking\n-# searching live courses\n-# thinking")
    );
    assert_ne!(
        thinking(&[], 2_000, 1),
        head,
        "the spinner turns with the frame"
    );

    let many: Vec<String> = (0..200).map(|i| format!("step number {i}")).collect();
    let folded = thinking(&many, 500, 0);
    assert!(folded.len() <= 500);
    assert!(folded.ends_with("step number 199"), "the newest step stays");
    assert!(folded.contains("earlier steps"), "{folded}");
}

#[test]
fn a_tool_result_writes_over_the_line_its_own_start_wrote() {
    use crate::render::card::Steps;

    let mut steps = Steps::default();
    assert!(steps.push(Some("model:1"), "thinking"));
    assert!(steps.push(Some("tool:c1"), "`search` running"));
    assert!(steps.push(Some("model:1"), "checking the hours page"));
    assert!(steps.push(Some("tool:c1"), "`search` gave 3 passages"));
    assert!(
        !steps.push(Some("tool:c1"), "`search` gave 3 passages"),
        "the same line twice changes nothing"
    );
    assert_eq!(
        steps.lines(),
        vec![
            "checking the hours page".to_owned(),
            "`search` gave 3 passages".to_owned()
        ],
        "each slot keeps its place and holds its newest line"
    );
}

#[test]
fn the_final_card_keeps_steps_then_answer_then_footers_in_order() {
    use crate::render::card::{THINKING, answer};

    let mut resp = response(
        "Hayden closes at 2am.",
        vec![linked("Hayden hours"), unlinked("desk note")],
        "answered",
    );
    resp.memories = vec!["You study CSE.".into()];
    let steps = vec![
        "\u{1f914} thinking".to_owned(),
        "\u{2705} `search_courses` \u{2192} three passages".to_owned(),
    ];

    let out = answer(&steps, &resp, 2_000);
    assert_eq!(out.len(), 1);
    let card = &out[0];
    assert!(!card.contains(THINKING), "the header goes once answered");
    let order = [
        "-# \u{1f914} thinking",
        "-# \u{2705} `search_courses`",
        "Hayden closes at 2am.",
        "Also from** desk note",
        "Memory used**\n- You study CSE.",
    ];
    let positions: Vec<usize> = order
        .iter()
        .map(|part| card.find(part).unwrap_or(usize::MAX))
        .collect();
    assert!(positions.iter().all(|&p| p != usize::MAX), "{card}");
    assert!(positions.windows(2).all(|w| w[0] < w[1]), "{card}");

    let bare = answer(&[], &response("hi", vec![], "answered"), 2_000);
    assert_eq!(bare, vec!["hi".to_owned()], "no footers without content");
}

#[test]
fn an_oversized_card_folds_steps_then_trims_footers_then_continues() {
    use crate::render::card::answer;

    let steps: Vec<String> = (0..40)
        .map(|i| format!("step {i} {}", "s".repeat(20)))
        .collect();
    let resp = response("short answer", vec![], "answered");
    let folded = answer(&steps, &resp, 500);
    assert_eq!(folded.len(), 1);
    assert!(folded[0].starts_with("-# 40 steps"), "{}", folded[0]);
    assert!(!folded[0].contains("step 0"), "{}", folded[0]);

    let mut resp = response("short answer", Vec::new(), "answered");
    resp.memories = (0..40)
        .map(|i| format!("memory {i} {}", "m".repeat(30)))
        .collect();
    let trimmed = answer(&steps, &resp, 500);
    assert_eq!(trimmed.len(), 1, "{trimmed:?}");
    assert!(trimmed[0].contains("- memory 2"));
    assert!(!trimmed[0].contains("- memory 3"));
    assert!(trimmed[0].contains("and 37 more"));

    let resp = response(&"word ".repeat(300), vec![linked("one")], "answered");
    let spilled = answer(&steps, &resp, 500);
    assert!(spilled.len() > 1, "only a long answer continues");
    assert!(spilled.iter().all(|m| m.len() <= 500));
    assert!(spilled[0].starts_with("-# 40 steps"));
}

#[test]
fn an_accepted_approval_keeps_the_steps_and_replaces_prompt_and_footers() {
    use crate::core::types::Confirmation;
    use crate::render::card::{answer, failed, resumed, steps_of};
    use crate::render::components::rows_for;

    let mut asked = response("", vec![unlinked("old source")], "awaiting_confirmation");
    asked.confirmation = Some(Confirmation {
        token: Uuid::new_v4(),
        tool: "announce".into(),
        summary: "Post the announcement.".into(),
    });
    let steps = vec!["thinking".to_owned(), "search finished".to_owned()];
    let card = answer(&steps, &asked, 2_000);
    assert_eq!(card.len(), 1);
    assert!(card[0].contains("`announce` needs your approval:** Post the announcement."));
    assert!(!card[0].contains("no answer"), "{}", card[0]);
    assert_eq!(labels_of(&rows_for(&asked)[0]), vec!["Yes, do it", "No"]);
    assert_eq!(steps_of(&card[0]), (0, steps.clone()));

    let done = response("Registered.", vec![unlinked("new source")], "answered");
    let out = resumed(&card[0], true, &done, 2_000);
    assert_eq!(
        out,
        vec![
            "-# thinking\n-# search finished\n-# approved\n\nRegistered.\n\n**\u{1f4da} Also from** new source"
                .to_owned()
        ]
    );
    let declined = resumed(&card[0], false, &done, 2_000);
    assert!(declined[0].contains("-# declined"));
    assert!(!declined[0].contains("needs your approval"));
    assert!(!declined[0].contains("old source"));

    let broke = failed(&["thinking".to_owned()], "Sparky is unavailable.", 2_000);
    assert_eq!(broke, "-# thinking\n\nSparky is unavailable.");
}

#[test]
fn a_resumed_card_at_the_discord_limit_folds_and_stays_within_it() {
    use crate::core::types::Confirmation;
    use crate::render::card::{answer, resumed, steps_of};

    let steps: Vec<String> = (0..60)
        .map(|i| format!("step {i} {}", "s".repeat(20)))
        .collect();
    let mut asked = response(&"a".repeat(400), vec![], "awaiting_confirmation");
    asked.confirmation = Some(Confirmation {
        token: Uuid::new_v4(),
        tool: "announce".into(),
        summary: "Post the announcement.".into(),
    });
    let card = answer(&steps, &asked, 2_000);
    assert_eq!(card.len(), 1);
    assert!(card[0].len() <= 2_000);
    assert!(card[0].starts_with("-# 60 steps"), "{}", card[0]);
    assert_eq!(steps_of(&card[0]).0, 60);

    let done = response(&"word ".repeat(380), vec![linked("one")], "answered");
    let out = resumed(&card[0], true, &done, 2_000);
    assert!(out.iter().all(|m| m.len() <= 2_000));
    assert!(
        out[0].starts_with("-# 60 steps\n-# approved\n\nword"),
        "{}",
        out[0]
    );

    let longer = response(&"word ".repeat(398), vec![linked("one")], "answered");
    let out = resumed(&card[0], true, &longer, 2_000);
    assert!(out.iter().all(|m| m.len() <= 2_000));
    assert!(out[0].starts_with("-# 61 steps\n\nword"), "{}", out[0]);
    assert!(!out[0].contains("needs your approval"));
}

#[test]
fn a_refused_press_is_told_privately_and_other_failures_read_as_outages() {
    use crate::core::types::EngineError;
    use crate::render::reply::{NOT_YOURS, UNAVAILABLE, confirm_failure, confirm_refused, failure};

    let status = |status| EngineError::Status {
        status,
        body: String::new(),
    };
    assert!(confirm_refused(&status(404)));
    assert!(confirm_refused(&status(409)));
    assert!(!confirm_refused(&status(502)));
    assert!(!confirm_refused(&EngineError::Transport("down".into())));
    assert_eq!(confirm_failure(&status(404)), NOT_YOURS);
    assert_eq!(confirm_failure(&status(500)), UNAVAILABLE);
    assert_eq!(failure(&status(409)), "That approval is no longer open.");
}

#[test]
fn the_pacer_holds_a_change_for_the_next_slot_and_never_drops_it() {
    use std::time::{Duration, Instant};

    use crate::render::card::Pacer;

    let every = Duration::from_millis(1_500);
    let start = Instant::now();
    let mut pacer = Pacer::new(every, start);
    assert_eq!(pacer.wait(start), None, "nothing waits");

    pacer.mark(false);
    assert_eq!(
        pacer.wait(start),
        None,
        "an unchanged step waits for nothing"
    );

    pacer.mark(true);
    let later = start + Duration::from_millis(500);
    assert_eq!(pacer.wait(later), Some(Duration::from_secs(1)));
    let late = start + Duration::from_secs(2);
    assert_eq!(
        pacer.wait(late),
        Some(Duration::ZERO),
        "overdue edits go now"
    );

    pacer.flushed(late);
    assert_eq!(pacer.wait(late), None);
    pacer.mark(true);
    assert_eq!(pacer.wait(late), Some(every));
}

#[test]
fn chunking_always_advances_even_at_tiny_limits() {
    let parts = chunk("ééé", 1);
    assert_eq!(parts, vec!["é", "é", "é"]);
    let parts = chunk("a é b", 2);
    assert_eq!(parts.concat().replace(' ', ""), "aéb");
}

#[test]
fn sse_frames_come_out_whole_even_when_the_bytes_arrive_split() {
    use crate::engine::sse::drain_frames;

    let mut buf = String::new();

    // A frame split across two reads yields nothing until it is complete.
    buf.push_str("event: progress\ndata: {\"text\":\"sear");
    assert!(drain_frames(&mut buf).is_empty());

    buf.push_str("ching ASU pages\"}\n\nevent: done\ndata: {}\n\n");
    let frames = drain_frames(&mut buf);
    assert_eq!(
        frames,
        vec![
            (
                "progress".to_owned(),
                "{\"text\":\"searching ASU pages\"}".to_owned()
            ),
            ("done".to_owned(), "{}".to_owned()),
        ]
    );
    assert!(buf.is_empty(), "consumed frames leave the buffer clean");
}

#[test]
fn a_frame_without_an_event_name_still_carries_its_data() {
    use crate::engine::sse::drain_frames;

    let mut buf = String::from("data: {\"a\":1}\n\n");
    assert_eq!(
        drain_frames(&mut buf),
        vec![(String::new(), "{\"a\":1}".to_owned())]
    );
}

#[test]
fn a_component_id_survives_the_round_trip_and_rejects_anything_else() {
    use uuid::Uuid;

    use crate::render::components::{Action, CustomId};

    let token = Uuid::new_v4();
    let convo = Uuid::new_v4();
    let id = CustomId::new(Action::Approve, token, convo);
    let wire = id.to_string();

    assert!(wire.starts_with("sparky:"), "{wire}");
    assert!(wire.len() <= 100, "Discord caps custom_id at 100 bytes");
    assert_eq!(CustomId::parse(&wire), Some(id));

    // Anything the bot did not mint is not ours to act on.
    assert_eq!(CustomId::parse("approve"), None);
    assert_eq!(CustomId::parse("other:approve:x:y"), None);
    assert_eq!(CustomId::parse("sparky:approve:not-a-uuid:x"), None);
    assert_eq!(CustomId::parse("sparky:launch:x:y"), None, "unknown action");
}

#[test]
fn a_confirmation_offers_the_two_answers_and_a_plain_answer_offers_none() {
    use uuid::Uuid;

    use crate::core::types::Confirmation;
    use crate::render::components::rows_for;

    let resp = response("", vec![], "awaiting_confirmation");
    assert!(rows_for(&resp).is_empty(), "no confirmation, no buttons");

    let mut asked = response("", vec![], "awaiting_confirmation");
    asked.confirmation = Some(Confirmation {
        token: Uuid::new_v4(),
        tool: "announce".into(),
        summary: "Post the announcement.".into(),
    });
    let rows = rows_for(&asked);
    assert_eq!(rows.len(), 1, "one row of answers");
    assert_eq!(rows[0].len(), 2, "approve and deny");
}

#[test]
fn forget_ids_carry_the_asker_and_survive_the_round_trip() {
    use crate::render::components::{CustomId, forget_rows};

    let user = 123_456_789_012_345_678_u64;
    for id in [CustomId::ForgetAll { user }, CustomId::KeepAll { user }] {
        let wire = id.to_string();
        assert!(wire.len() <= 100, "{wire}");
        assert_eq!(CustomId::parse(&wire), Some(id));
    }
    assert_eq!(CustomId::parse("sparky:forget_all:not-a-number"), None);
    assert_eq!(CustomId::parse("sparky:forget_all:1:extra"), None);
    assert_eq!(CustomId::parse("sparky:forget_all"), None);

    let rows = forget_rows(user);
    assert_eq!(
        rows[0][0],
        crate::render::components::ButtonSpec::Press {
            id: CustomId::ForgetAll { user },
            label: "Forget everything",
            danger: true,
        }
    );
    assert_eq!(labels_of(&rows[0]), vec!["Forget everything", "Cancel"]);

    assert!(CustomId::ForgetAll { user }.may_press(user));
    assert!(!CustomId::ForgetAll { user }.may_press(user + 1));
    assert!(!CustomId::KeepAll { user }.may_press(7));
    let confirm = CustomId::new(
        crate::render::components::Action::Deny,
        Uuid::new_v4(),
        Uuid::new_v4(),
    );
    assert!(
        confirm.may_press(7),
        "the engine checks confirmation callers"
    );

    let lost = crate::render::reply::memory_failure(&crate::core::types::EngineError::Status {
        status: 404,
        body: String::new(),
    });
    assert_eq!(lost, crate::render::reply::UNAVAILABLE);
}

#[test]
fn mentions_of_the_bot_are_stripped_and_others_kept() {
    use crate::access::route::strip_mentions;
    use serenity::all::UserId;

    let me = UserId::new(42);
    assert_eq!(
        strip_mentions("<@42> when does Hayden close?", me),
        "when does Hayden close?"
    );
    assert_eq!(strip_mentions("hey <@!42> ask <@7>", me), "hey   ask <@7>");
    assert_eq!(strip_mentions("  <@42> <@!42> ", me), "");
}

#[test]
fn permissions_are_the_union_of_everyone_and_held_roles() {
    use std::collections::HashMap;

    use crate::access::roles::member_permissions;
    use serenity::all::RoleId;

    let everyone = RoleId::new(1);
    let mods = RoleId::new(2);
    let admins = RoleId::new(3);
    let bits = HashMap::from([
        (everyone, Permissions::SEND_MESSAGES),
        (mods, Permissions::MANAGE_MESSAGES),
        (admins, Permissions::MANAGE_GUILD),
    ]);

    let plain = member_permissions(everyone, &[], &bits, false);
    assert_eq!(plain, Permissions::SEND_MESSAGES);
    assert!(!can_write(plain));

    let moderator = member_permissions(everyone, &[mods], &bits, false);
    assert!(moderator.contains(Permissions::SEND_MESSAGES | Permissions::MANAGE_MESSAGES));
    assert!(!can_write(moderator));

    let admin = member_permissions(everyone, &[mods, admins, RoleId::new(99)], &bits, false);
    assert!(
        can_write(admin),
        "a held role grants write; an unknown role adds nothing"
    );

    let owner = member_permissions(everyone, &[], &bits, true);
    assert!(can_write(owner), "the guild owner writes without any role");
}

#[test]
fn only_threads_inherit_the_allowlist_of_their_parent() {
    use crate::access::route::{serves, thread_parent};
    use serenity::all::{ChannelId, ChannelType};

    let category = ChannelId::new(10);
    let channel = ChannelId::new(11);
    let allow = [category];

    let text_parent = thread_parent(Some(ChannelType::Text), Some(category));
    assert_eq!(text_parent, None, "a category is not a thread parent");
    assert!(!serves(&allow, channel, text_parent));

    let thread_parent_id = thread_parent(Some(ChannelType::PublicThread), Some(category));
    assert_eq!(thread_parent_id, Some(category));
    assert!(serves(&allow, ChannelId::new(12), thread_parent_id));

    assert!(serves(&[], channel, None), "an empty allowlist admits all");
    assert!(serves(&[channel], channel, None));
    assert_eq!(thread_parent(None, Some(category)), None);
}

#[test]
fn an_ephemeral_press_confirms_privately_and_stays_ephemeral() {
    use crate::access::route::press_visibility;
    use crate::core::types::Visibility;
    use serenity::all::MessageFlags;

    assert_eq!(
        press_visibility(Some(MessageFlags::EPHEMERAL)),
        (Visibility::Private, true)
    );
    assert_eq!(
        press_visibility(Some(MessageFlags::empty())),
        (Visibility::Public, false)
    );
    assert_eq!(press_visibility(None), (Visibility::Public, false));
}

#[test]
fn thread_names_fit_discord_and_are_never_empty() {
    use crate::access::route::{THREAD_NAME_MAX, thread_name};

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
fn the_asked_header_stays_within_the_limit() {
    use crate::access::route::asked_header;

    assert_eq!(asked_header("Ash", "hi", 2_000), "**Ash asked:** hi");
    let long = asked_header("Ash", &"x".repeat(5_000), 2_000);
    assert_eq!(long.chars().count(), 2_000);
    assert!(long.ends_with("..."));
}

#[test]
fn each_place_sets_visibility_and_continuation() {
    use crate::access::route::{AskMode, Place, ask_mode, chat_request};
    use crate::core::types::Visibility;
    use serenity::all::{ChannelId, ChannelType, GuildId, UserId};

    let at = ChannelId::new(5);
    let make = |place| chat_request(place, UserId::new(1), GuildId::new(2), vec![], "q".into());

    let private = make(Place::Private(at));
    assert_eq!(private.visibility, Visibility::Private);
    assert!(private.continue_channel);

    let thread = make(Place::Thread(at));
    assert_eq!(thread.visibility, Visibility::Public);
    assert!(thread.continue_channel);
    assert_eq!(thread.channel_id, "5");
    assert!(thread.conversation_id.is_none());

    let inline = make(Place::Inline(at));
    assert_eq!(inline.visibility, Visibility::Public);
    assert!(!inline.continue_channel);

    let wire = serde_json::to_value(&private).unwrap_or_default();
    assert_eq!(wire["visibility"], "private");
    assert_eq!(wire["continue_channel"], true);

    assert_eq!(
        ask_mode(true, Some(ChannelType::PublicThread)),
        AskMode::Private
    );
    assert_eq!(
        ask_mode(false, Some(ChannelType::PrivateThread)),
        AskMode::InThread
    );
    assert_eq!(
        ask_mode(false, Some(ChannelType::NewsThread)),
        AskMode::InThread
    );
    assert_eq!(ask_mode(false, Some(ChannelType::Text)), AskMode::NewThread);
    assert_eq!(ask_mode(false, None), AskMode::NewThread);
}

#[test]
fn memory_renders_things_and_relations_or_says_it_is_empty() {
    use crate::core::types::{EngineError, ProfileList, ProfileNode, ProfileRelation};
    use crate::render::reply::{NOTHING_REMEMBERED, forgot, memory_failure, render_profile};

    assert_eq!(
        render_profile(&ProfileList::default(), 2_000),
        vec![NOTHING_REMEMBERED.to_owned()]
    );

    let profile = ProfileList {
        nodes: vec![ProfileNode {
            kind: "course".into(),
            label: "CSE 310".into(),
            confidence: 0.87,
        }],
        relations: vec![ProfileRelation {
            subject: "me".into(),
            relation: "enrolled_in".into(),
            object: "CSE 310".into(),
            confidence: 0.5,
        }],
    };
    let out = render_profile(&profile, 2_000).join("\n");
    assert!(out.contains("- CSE 310 (course, 87%)"), "{out}");
    assert!(out.contains("- me enrolled in CSE 310 (50%)"), "{out}");

    let many = ProfileList {
        nodes: (0..200)
            .map(|i| ProfileNode {
                kind: "club".into(),
                label: format!("club number {i}"),
                confidence: 1.0,
            })
            .collect(),
        relations: vec![],
    };
    let parts = render_profile(&many, 500);
    assert!(parts.len() > 1);
    assert!(parts.iter().all(|p| p.len() <= 500));

    assert_eq!(forgot(0, true), "I had nothing under that name.");
    assert_eq!(forgot(3, false), "Forgot 3 things.");
    let off = memory_failure(&EngineError::Status {
        status: 503,
        body: String::new(),
    });
    assert_eq!(off, "Memory is turned off here.");
}

#[test]
fn an_analytics_event_serializes_to_the_posthog_batch_shape() {
    use crate::core::types::{AnalyticsBatch, AnalyticsEvent};

    let event = AnalyticsEvent::new("discord_ask", &42_u64)
        .with("place", "thread")
        .with("latency_ms", 120);
    let batch = [event];
    let wire = serde_json::to_value(AnalyticsBatch {
        api_key: "phc_test",
        batch: &batch,
    })
    .unwrap_or_default();
    assert_eq!(wire["api_key"], "phc_test");
    let first = &wire["batch"][0];
    assert_eq!(first["event"], "discord_ask");
    assert_eq!(first["distinct_id"], "42");
    assert_eq!(first["properties"]["place"], "thread");
    assert_eq!(first["properties"]["latency_ms"], 120);
    let stamp = first["timestamp"].as_str().unwrap_or_default();
    assert!(stamp.ends_with('Z') && stamp.contains('.'), "{stamp}");
}

#[test]
fn a_full_analytics_queue_drops_instead_of_waiting() {
    use crate::analytics::Analytics;
    use crate::core::types::AnalyticsEvent;

    let (handle, mut rx) = Analytics::channel(1);
    assert!(handle.record(AnalyticsEvent::new("a", &1_u64)));
    assert!(!handle.record(AnalyticsEvent::new("b", &1_u64)), "full");
    assert_eq!(rx.try_recv().map(|e| e.event).ok(), Some("a"));
    drop(rx);
    assert!(!handle.record(AnalyticsEvent::new("c", &1_u64)), "closed");
    assert!(!Analytics::disabled().record(AnalyticsEvent::new("d", &1_u64)));
}

#[test]
fn analytics_stays_off_without_the_switch_or_a_project_token() {
    use crate::analytics::Analytics;
    use crate::core::config::{Analytics as Settings, Telemetry};
    use crate::core::types::AnalyticsEvent;

    let off = Settings {
        enabled: false,
        ..Settings::default()
    };
    let (handle, flusher) = Analytics::start(&off, &Telemetry::default());
    assert!(flusher.is_none());
    assert!(!handle.record(AnalyticsEvent::new("a", &1_u64)));

    let (handle, flusher) = Analytics::start(&Settings::default(), &Telemetry::default());
    assert!(flusher.is_none(), "the default token is empty");
    assert!(!handle.record(AnalyticsEvent::new("a", &1_u64)));
}

#[test]
fn the_export_target_needs_a_host_and_a_token_and_trims_the_slash() {
    use crate::core::config::Telemetry;
    use crate::core::telemetry::export_target;
    use secrecy::{ExposeSecret, SecretString};

    let mut cfg = Telemetry::default();
    assert!(export_target(&cfg).is_none(), "empty token");
    cfg.project_token = SecretString::from("phc_x".to_owned());
    cfg.host = Some(" http://posthog:8000/ ".into());
    let target = export_target(&cfg).map(|(h, t)| (h.to_owned(), t.expose_secret().to_owned()));
    assert_eq!(
        target,
        Some(("http://posthog:8000".to_owned(), "phc_x".to_owned()))
    );
    cfg.host = Some("  ".into());
    assert!(export_target(&cfg).is_none(), "blank host");
    cfg.host = None;
    assert!(export_target(&cfg).is_none());
}

#[test]
fn analytics_settings_reject_an_empty_queue() {
    use crate::core::config::Analytics;

    assert!(Analytics::default().validate().is_ok());
    let bad = Analytics {
        queue_capacity: 0,
        ..Analytics::default()
    };
    assert!(bad.validate().is_err());
}

/// Requests a local server received: path, authorization header, body.
type Seen = std::sync::Arc<std::sync::Mutex<Vec<(String, String, String)>>>;

async fn record_request(
    axum::extract::State(seen): axum::extract::State<Seen>,
    uri: axum::http::Uri,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> axum::http::StatusCode {
    let auth = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_owned();
    let body = String::from_utf8_lossy(&body).into_owned();
    if let Ok(mut seen) = seen.lock() {
        seen.push((uri.path().to_owned(), auth, body));
    }
    axum::http::StatusCode::OK
}

/// Serves on a thread of its own and returns the bound address.
fn serve(seen: Seen) -> Option<std::net::SocketAddr> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let Ok(rt) = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        else {
            return;
        };
        rt.block_on(async move {
            let Ok(listener) = tokio::net::TcpListener::bind("127.0.0.1:0").await else {
                return;
            };
            let _ = tx.send(listener.local_addr().ok());
            let app = axum::Router::new()
                .fallback(record_request)
                .with_state(seen);
            let _ = axum::serve(listener, app).await;
        });
    });
    rx.recv_timeout(std::time::Duration::from_secs(5))
        .ok()
        .flatten()
}

/// What the server received once n requests arrived, or at the deadline.
fn wait_for(seen: &Seen, n: usize) -> Vec<(String, String, String)> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let mut got = Vec::new();
    while std::time::Instant::now() < deadline {
        got = seen.lock().map(|s| s.clone()).unwrap_or_default();
        if got.len() >= n {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    got
}

#[test]
fn analytics_batches_reach_the_batch_path_with_the_api_key() {
    use crate::analytics::Analytics;
    use crate::core::config::{Analytics as Settings, Telemetry};
    use crate::core::types::AnalyticsEvent;
    use secrecy::SecretString;

    let seen = Seen::default();
    let addr = serve(std::sync::Arc::clone(&seen));
    assert!(addr.is_some());
    let Some(addr) = addr else {
        return;
    };
    let telemetry = Telemetry {
        host: Some(format!("http://{addr}/")),
        project_token: SecretString::from("phc_test".to_owned()),
        ..Telemetry::default()
    };
    let settings = Settings {
        flush_ms: 50,
        ..Settings::default()
    };
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build();
    assert!(rt.is_ok());
    let Ok(rt) = rt else {
        return;
    };
    rt.block_on(async {
        let (handle, flusher) = Analytics::start(&settings, &telemetry);
        assert!(handle.record(AnalyticsEvent::new("discord_ask", &42_u64).with("place", "thread")));
        assert!(flusher.is_some());
        if let Some(flusher) = flusher {
            flusher.finish(std::time::Duration::from_secs(5)).await;
        }
    });

    let got = wait_for(&seen, 1);
    assert_eq!(got.len(), 1, "{got:?}");
    let Some((path, _, body)) = got.first() else {
        return;
    };
    assert_eq!(path, "/batch/");
    let wire: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
    assert_eq!(wire["api_key"], "phc_test");
    assert_eq!(wire["batch"][0]["event"], "discord_ask");
    assert_eq!(wire["batch"][0]["distinct_id"], "42");
    assert_eq!(wire["batch"][0]["properties"]["place"], "thread");
}

#[test]
fn discord_spans_reach_the_traces_path_with_the_bearer_token() {
    use crate::core::config::Telemetry;
    use crate::core::telemetry::provider;
    use opentelemetry::trace::{Span as _, Tracer as _, TracerProvider as _};
    use secrecy::SecretString;

    let seen = Seen::default();
    let addr = serve(std::sync::Arc::clone(&seen));
    assert!(addr.is_some());
    let Some(addr) = addr else {
        return;
    };
    let cfg = Telemetry {
        host: Some(format!("http://{addr}")),
        project_token: SecretString::from("phc_test".to_owned()),
        // Off by default; set here to prove it is still reachable when it is configured.
        ai_path: "/i/v0/ai/otel".into(),
        ..Telemetry::default()
    };
    let built = provider(&cfg, "discord-test", "test");
    assert!(built.as_ref().is_ok_and(Option::is_some), "{built:?}");
    let Ok(Some(provider)) = built else {
        return;
    };
    let mut span = provider.tracer("discord-test").start("probe");
    span.end();
    let flushed = provider.force_flush();
    assert!(flushed.is_ok(), "{flushed:?}");

    let got = wait_for(&seen, 2);
    let _ = provider.shutdown();
    let mut paths: Vec<&str> = got.iter().map(|(p, _, _)| p.as_str()).collect();
    paths.sort_unstable();
    assert_eq!(paths, ["/i/v0/ai/otel", "/i/v1/traces"]);
    assert!(
        got.iter().all(|(_, auth, _)| auth == "Bearer phc_test"),
        "{got:?}"
    );
}

#[test]
fn the_ai_path_is_off_unless_it_is_configured() {
    use crate::core::config::Telemetry;

    // The self-hosted capture-ai service refuses OTLP, so a batch sent there fails forever.
    assert!(Telemetry::default().ai_path.is_empty());
    assert!(Telemetry::default().validate().is_ok());
}

#[test]
fn discord_spans_reach_phoenix_without_a_token() {
    use crate::core::config::Telemetry;
    use crate::core::telemetry::{phoenix_target, provider};
    use opentelemetry::trace::{Span as _, Tracer as _, TracerProvider as _};

    let seen = Seen::default();
    let addr = serve(std::sync::Arc::clone(&seen));
    assert!(addr.is_some());
    let Some(addr) = addr else {
        return;
    };
    // PostHog off, Phoenix on: the two destinations are independent.
    let cfg = Telemetry {
        host: None,
        phoenix_url: Some(format!(" http://{addr}/ ")),
        ..Telemetry::default()
    };
    assert_eq!(
        phoenix_target(&cfg),
        Some(format!("http://{addr}/v1/traces"))
    );
    let built = provider(&cfg, "discord-test", "test");
    assert!(built.as_ref().is_ok_and(Option::is_some), "{built:?}");
    let Ok(Some(provider)) = built else {
        return;
    };
    let mut span = provider.tracer("discord-test").start("probe");
    span.end();
    let flushed = provider.force_flush();
    assert!(flushed.is_ok(), "{flushed:?}");

    let got = wait_for(&seen, 1);
    let _ = provider.shutdown();
    assert_eq!(
        got.iter().map(|(p, _, _)| p.as_str()).collect::<Vec<_>>(),
        ["/v1/traces"]
    );
    assert!(got.iter().all(|(_, auth, _)| auth.is_empty()), "{got:?}");
}

#[test]
fn export_is_off_only_when_every_destination_is_unset() {
    use crate::core::config::Telemetry;
    use crate::core::telemetry::provider;

    let off = Telemetry {
        host: None,
        phoenix_url: None,
        ..Telemetry::default()
    };
    assert!(matches!(provider(&off, "d", "test"), Ok(None)));
    let phoenix_only = Telemetry {
        host: None,
        phoenix_url: Some("http://localhost:6006".into()),
        ..Telemetry::default()
    };
    assert!(matches!(provider(&phoenix_only, "d", "test"), Ok(Some(_))));
}
