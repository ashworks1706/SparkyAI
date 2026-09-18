//! Pure helpers of the Postgres store: fusion and vector encoding.

use uuid::Uuid;

use crate::stores::postgres::{rrf, vector_literal};

#[test]
fn vector_literal_matches_pgvector_input() {
    assert_eq!(vector_literal(&[0.5, -1.0, 2.25]), "[0.5,-1,2.25]");
}

#[test]
fn rrf_prefers_items_ranked_by_both_lists() {
    let a = Uuid::new_v4();
    let b = Uuid::new_v4();
    let c = Uuid::new_v4();
    let fused = rrf(&[vec![a, b], vec![b, c]], 60.0);
    assert_eq!(fused.first().map(|f| f.0), Some(b));
}

#[test]
fn a_chunk_its_own_summary_already_covers_is_dropped() {
    use uuid::Uuid;

    use crate::stores::postgres::collapse;

    let parent = Uuid::new_v4();
    let child = Uuid::new_v4();
    let other = Uuid::new_v4();

    // The summary ranks above its chunk, so the chunk is dropped.
    let keep = collapse(&[(parent, None), (child, Some(parent)), (other, None)]);
    assert_eq!(keep, vec![true, false, true]);
}

#[test]
fn a_chunk_that_outranks_its_summary_keeps_them_both() {
    use uuid::Uuid;

    use crate::stores::postgres::collapse;

    let parent = Uuid::new_v4();
    let child = Uuid::new_v4();

    // The chunk ranks above its summary, so both are kept.
    let keep = collapse(&[(child, Some(parent)), (parent, None)]);
    assert_eq!(keep, vec![true, true]);
}

#[test]
fn a_flat_index_is_left_alone() {
    use uuid::Uuid;

    use crate::stores::postgres::collapse;

    // Rows with no parent are all kept.
    let rows: Vec<(Uuid, Option<Uuid>)> = (0..5).map(|_| (Uuid::new_v4(), None)).collect();
    assert_eq!(collapse(&rows), vec![true; 5]);
}

#[test]
fn a_query_deadline_with_nanoseconds_is_kept_to_microseconds() {
    use crate::stores::knowledge::query::deadline_interval;

    let remaining = std::time::Duration::from_nanos(88_100_127_552);
    let Ok(interval) = deadline_interval(remaining) else {
        unreachable!("a deadline with nanoseconds is a valid interval")
    };
    assert_eq!(interval.microseconds, 88_100_127);
    assert_eq!((interval.months, interval.days), (0, 0));
}

#[test]
fn the_wait_between_looks_at_a_queued_job_doubles_up_to_its_cap() {
    use std::time::Duration;

    use crate::stores::knowledge::query::backoff;

    let most = Duration::from_secs(1);
    let mut wait = Duration::from_millis(100);
    let mut waits = vec![wait];
    for _ in 0..6 {
        wait = backoff(wait, most);
        waits.push(wait);
    }
    assert_eq!(
        waits.iter().map(Duration::as_millis).collect::<Vec<_>>(),
        [100, 200, 400, 800, 1000, 1000, 1000]
    );

    // A 30 second fetch is looked at far fewer times than a fixed 100ms poll would look.
    let mut elapsed = Duration::ZERO;
    let mut looks = 0;
    let mut wait = Duration::from_millis(100);
    while elapsed < Duration::from_secs(30) {
        elapsed += wait;
        looks += 1;
        wait = backoff(wait, most);
    }
    assert!(looks < 40, "{looks} looks, a fixed poll would take 300");
}
