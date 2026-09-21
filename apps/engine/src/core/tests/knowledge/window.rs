//! Sentence windows: the runs a hit reads back with, and the text they join into.

use std::collections::HashMap;

use uuid::Uuid;

use crate::stores::knowledge::retrieval::{Span, locate, spans};

#[test]
fn a_hit_reads_back_with_its_neighbours() {
    let page = Uuid::new_v4();
    assert_eq!(
        spans(&[(page, 7)], 2),
        vec![Span {
            version_id: page,
            lo: 5,
            hi: 9
        }]
    );
}

#[test]
fn a_window_of_zero_hands_back_the_hit_alone() {
    let page = Uuid::new_v4();
    assert_eq!(
        spans(&[(page, 7)], 0),
        vec![Span {
            version_id: page,
            lo: 7,
            hi: 7
        }]
    );
}

#[test]
fn a_hit_at_the_start_of_a_page_does_not_reach_before_it() {
    let page = Uuid::new_v4();
    assert_eq!(spans(&[(page, 1)], 4).first().map(|s| s.lo), Some(0));
}

#[test]
fn two_hits_a_row_apart_are_one_passage() {
    let page = Uuid::new_v4();
    // 0..4 and 2..6 overlap and merge into 0..6.
    assert_eq!(
        spans(&[(page, 2), (page, 4)], 2),
        vec![Span {
            version_id: page,
            lo: 0,
            hi: 6
        }]
    );
}

#[test]
fn two_windows_that_only_touch_still_merge() {
    let page = Uuid::new_v4();
    // 0..2 and 3..5 share no row but are adjacent: one run of text, not two passages.
    assert_eq!(
        spans(&[(page, 1), (page, 4)], 1),
        vec![Span {
            version_id: page,
            lo: 0,
            hi: 5
        }]
    );
}

#[test]
fn two_hits_far_apart_on_one_page_stay_separate() {
    let page = Uuid::new_v4();
    assert_eq!(
        // Given newest first, they come back in reading order.
        spans(&[(page, 40), (page, 2)], 2),
        vec![
            Span {
                version_id: page,
                lo: 0,
                hi: 4
            },
            Span {
                version_id: page,
                lo: 38,
                hi: 42
            }
        ]
    );
}

#[test]
fn the_same_ordinal_on_two_pages_does_not_merge() {
    let one = Uuid::new_v4();
    let two = Uuid::new_v4();
    let merged = spans(&[(one, 5), (two, 5)], 2);
    assert_eq!(merged.len(), 2);
    assert_ne!(merged[0].version_id, merged[1].version_id);
}

#[test]
fn no_hits_read_back_nothing() {
    assert!(spans(&[], 2).is_empty());
}

#[test]
fn a_span_joins_its_rows_in_ordinal_order() {
    let page = Uuid::new_v4();
    let mut rows = HashMap::new();
    // Inserted out of order: the join follows the ordinal, not the order rows came back in.
    rows.insert((page, 6), "third".to_owned());
    rows.insert((page, 4), "first".to_owned());
    rows.insert((page, 5), "second".to_owned());
    let span = Span {
        version_id: page,
        lo: 4,
        hi: 6,
    };
    assert_eq!(span.join(&rows), "first second third");
}

#[test]
fn a_span_running_past_the_end_of_a_page_joins_what_is_there() {
    let page = Uuid::new_v4();
    let mut rows = HashMap::new();
    rows.insert((page, 0), "only".to_owned());
    let span = Span {
        version_id: page,
        lo: 0,
        hi: 4,
    };
    assert_eq!(span.join(&rows), "only");
}

#[test]
fn a_span_joins_no_rows_of_another_page() {
    let page = Uuid::new_v4();
    let other = Uuid::new_v4();
    let mut rows = HashMap::new();
    rows.insert((other, 1), "elsewhere".to_owned());
    let span = Span {
        version_id: page,
        lo: 0,
        hi: 2,
    };
    assert_eq!(span.join(&rows), "");
}

#[test]
fn a_hit_finds_the_span_that_holds_it() {
    let page = Uuid::new_v4();
    let other = Uuid::new_v4();
    let built = spans(&[(page, 2), (page, 40), (other, 2)], 2);
    let Some(first) = locate(&built, page, 2) else {
        unreachable!("the hit is inside a span built from it")
    };
    let Some(second) = locate(&built, page, 40) else {
        unreachable!("the hit is inside a span built from it")
    };
    assert_ne!(first, second);
    assert_eq!(locate(&built, page, 3), Some(first));
    assert_ne!(locate(&built, other, 2), Some(first));
}

#[test]
fn the_first_and_last_rows_of_a_span_are_inside_it() {
    let page = Uuid::new_v4();
    let built = spans(&[(page, 5)], 2);
    assert_eq!(locate(&built, page, 3), Some(0));
    assert_eq!(locate(&built, page, 7), Some(0));
    assert_eq!(locate(&built, page, 2), None);
    assert_eq!(locate(&built, page, 8), None);
}

#[test]
fn a_row_no_span_covers_is_not_found() {
    let page = Uuid::new_v4();
    let built = spans(&[(page, 2)], 2);
    assert_eq!(locate(&built, page, 9), None);
    assert_eq!(locate(&built, Uuid::new_v4(), 2), None);
}

#[test]
fn a_span_two_hits_landed_in_is_handed_back_once() {
    use crate::stores::knowledge::retrieval::{Take, takes};

    let page = Uuid::new_v4();
    let built = spans(&[(page, 2), (page, 4)], 2);
    let rows = [(0, page, 2), (0, page, 4)];
    assert_eq!(takes(&rows, &built), vec![Take::Span(0), Take::Skip]);
}

#[test]
fn two_spans_of_one_page_are_both_handed_back() {
    use crate::stores::knowledge::retrieval::{Take, takes};

    let page = Uuid::new_v4();
    let built = spans(&[(page, 2), (page, 40)], 2);
    let rows = [(0, page, 40), (0, page, 2)];
    // Ranked with the later passage first, the ranking is what decides the order.
    assert_eq!(takes(&rows, &built), vec![Take::Span(1), Take::Span(0)]);
}

#[test]
fn a_summary_keeps_its_own_text() {
    use crate::stores::knowledge::retrieval::{Take, takes};

    let page = Uuid::new_v4();
    let built = spans(&[(page, 9)], 2);
    // Summaries are ordinalled after the leaves, so a span off the end of a page reaches one.
    // The summary is not widened.
    let rows = [(1, page, 10), (0, page, 9)];
    assert_eq!(takes(&rows, &built), vec![Take::Own, Take::Span(0)]);
}

#[test]
fn a_chunk_no_span_covers_keeps_its_own_text() {
    use crate::stores::knowledge::retrieval::{Take, takes};

    let page = Uuid::new_v4();
    let rows = [(0, page, 2)];
    assert_eq!(takes(&rows, &[]), vec![Take::Own]);
}
