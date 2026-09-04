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
