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

    // Fused order is best first. The summary outranks its chunk, so the chunk is covered.
    let keep = collapse(&[(parent, None), (child, Some(parent)), (other, None)]);
    assert_eq!(keep, vec![true, false, true]);
}

#[test]
fn a_chunk_that_outranks_its_summary_keeps_them_both() {
    use uuid::Uuid;

    use crate::stores::postgres::collapse;

    let parent = Uuid::new_v4();
    let child = Uuid::new_v4();

    // The chunk scored higher, so it is the specific answer and the summary is the context.
    let keep = collapse(&[(child, Some(parent)), (parent, None)]);
    assert_eq!(keep, vec![true, true]);
}

#[test]
fn a_flat_index_is_left_alone() {
    use uuid::Uuid;

    use crate::stores::postgres::collapse;

    // Nothing has a parent until a tree is built, so collapsing must change nothing.
    let rows: Vec<(Uuid, Option<Uuid>)> = (0..5).map(|_| (Uuid::new_v4(), None)).collect();
    assert_eq!(collapse(&rows), vec![true; 5]);
}
