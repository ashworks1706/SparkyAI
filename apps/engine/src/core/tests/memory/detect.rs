//! The gate on profile extraction. Rules, so it costs no model call on a greeting.

use crate::agent::harness::memory::detect::{RuleDetector, Rules};
use crate::core::config;
use crate::core::traits::memory::detector::FactDetector;

fn detector() -> RuleDetector {
    RuleDetector::default()
}

#[test]
fn a_stated_preference_passes_the_gate() {
    let d = detector();
    for turn in [
        "I prefer studying at Hayden Library in the evenings",
        "I am a computer science major",
        "I'm taking CSE 310 this term",
        "My name is Alex and I live in Tempe",
        "I joined the robotics club last semester",
        "I usually work on weekends",
    ] {
        assert!(d.carries_fact(turn), "{turn:?} states something");
    }
}

#[test]
fn a_negated_preference_still_passes() {
    // This is why the gate is rules. Generic embeddings place I like this and I do not like
    // this close together, so a cosine gate cannot tell them apart. Polarity is extraction's
    // problem, and it only gets the chance if the gate lets the turn through.
    let d = detector();
    assert!(d.carries_fact("I do not like studying in the library"));
    assert!(d.carries_fact("I never work on weekends"));
}

#[test]
fn small_talk_and_questions_cost_nothing() {
    let d = detector();
    for turn in [
        "hi",
        "hello there",
        "thanks!",
        "thank you so much",
        "when does Hayden Library close?",
        "what clubs are there?",
        "Can you tell me about ASU scholarships?",
        "ok",
        "",
        "   ",
    ] {
        assert!(
            !d.carries_fact(turn),
            "{turn:?} states nothing about the asker"
        );
    }
}

#[test]
fn a_question_about_oneself_is_still_a_question() {
    // I am asking, not stating. A question ends in a question mark whatever else it carries.
    let d = detector();
    assert!(!d.carries_fact("What classes should I take as a computer science major?"));
}

#[test]
fn a_sentence_with_no_first_person_subject_is_not_about_the_asker() {
    let d = detector();
    assert!(!d.carries_fact("Hayden Library is open until 2am on weekdays"));
    assert!(!d.carries_fact("The robotics club meets on Thursdays"));
}

#[test]
fn a_marker_inside_another_word_does_not_count() {
    // Without word boundaries, imagine and limited would both look like a first person I.
    let d = detector();
    assert!(!d.carries_fact("Imagine the library was always open"));
    assert!(!d.carries_fact("Financial aid is limited this year"));
}

#[test]
fn the_rules_come_from_configuration() {
    let cfg = config::Detector::default();
    let rules = Rules::from(&cfg);
    assert_eq!(rules.min_words, cfg.min_words);
    assert!(
        !rules.subjects.is_empty(),
        "an empty list falls back to the built-in one"
    );
    assert!(!rules.cues.is_empty());

    // A deployment can narrow the gate to what it cares about.
    let narrow = RuleDetector::new(Rules {
        subjects: vec!["i ".into()],
        cues: vec!["major".into()],
        min_words: 3,
    });
    assert!(narrow.carries_fact("I am a physics major"));
    assert!(!narrow.carries_fact("I prefer mornings"));
}

#[test]
fn a_turn_shorter_than_the_floor_is_skipped() {
    let d = RuleDetector::new(Rules {
        min_words: 5,
        ..Rules::default()
    });
    assert!(!d.carries_fact("I like cats"));
    assert!(d.carries_fact("I like cats and long walks"));
}
