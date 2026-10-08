//! The arithmetic of retrieval: read-back windows around hits, and rank fusion of the legs.

use std::collections::{HashMap, HashSet};

use uuid::Uuid;

/// One run of rows a hit reads back with, and the hit that earned it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Span {
    /// The source version the run belongs to.
    pub version_id: Uuid,
    /// First ordinal of the run.
    pub lo: i32,
    /// Last ordinal of the run.
    pub hi: i32,
}

impl Span {
    /// The run text, its rows in ordinal order separated by a space, absent rows skipped.
    pub(crate) fn join(&self, rows: &HashMap<(Uuid, i32), String>) -> String {
        let mut parts: Vec<&str> = Vec::new();
        for ordinal in self.lo..=self.hi {
            if let Some(text) = rows.get(&(self.version_id, ordinal)) {
                parts.push(text);
            }
        }
        parts.join(" ")
    }
}

/// The spans hits read back with, overlapping ones merged.
pub(crate) fn spans(hits: &[(Uuid, i32)], window: i32) -> Vec<Span> {
    let mut wanted: Vec<Span> = hits
        .iter()
        .map(|(version_id, ordinal)| Span {
            version_id: *version_id,
            lo: ordinal.saturating_sub(window).max(0),
            hi: ordinal.saturating_add(window),
        })
        .collect();
    wanted.sort_by_key(|s| (s.version_id, s.lo));

    let mut merged: Vec<Span> = Vec::with_capacity(wanted.len());
    for span in wanted {
        match merged.last_mut() {
            // Touching counts as overlapping: an adjacent run is one passage, not two.
            Some(last)
                if last.version_id == span.version_id && span.lo <= last.hi.saturating_add(1) =>
            {
                last.hi = last.hi.max(span.hi);
            }
            _ => merged.push(span),
        }
    }
    merged
}

/// The index of the span holding a row, if one does.
pub(crate) fn locate(spans: &[Span], version_id: Uuid, ordinal: i32) -> Option<usize> {
    spans
        .iter()
        .position(|s| s.version_id == version_id && s.lo <= ordinal && ordinal <= s.hi)
}

/// What a ranked row contributes to the evidence handed back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Take {
    /// Dropped: a higher ranked row already carries this span.
    Skip,
    /// Kept with its own text.
    Own,
    /// Kept with the text of the span at this index.
    Span(usize),
}

/// What each ranked row contributes, highest ranked first.
pub(crate) fn takes(rows: &[(i32, Uuid, i32)], spans: &[Span]) -> Vec<Take> {
    let mut seen: HashSet<usize> = HashSet::new();
    rows.iter()
        .map(|(level, version_id, ordinal)| {
            if *level != 0 {
                return Take::Own;
            }
            match locate(spans, *version_id, *ordinal) {
                Some(at) if seen.insert(at) => Take::Span(at),
                Some(_) => Take::Skip,
                None => Take::Own,
            }
        })
        .collect()
}

/// Drops a row whose summary is already in the result, keeping the higher ranked of the two.
pub(crate) fn collapse(rows: &[(Uuid, Option<Uuid>)]) -> Vec<bool> {
    let mut seen: HashSet<Uuid> = HashSet::new();
    rows.iter()
        .map(|(id, parent)| {
            if parent.is_some_and(|p| seen.contains(&p)) {
                return false;
            }
            seen.insert(*id);
            true
        })
        .collect()
}

/// Reciprocal rank fusion. Each ranked list contributes 1 / (k + rank).
pub(crate) fn rrf(lists: &[Vec<Uuid>], k: f32) -> Vec<(Uuid, f32)> {
    let mut scores: HashMap<Uuid, f32> = HashMap::new();
    for list in lists {
        for (rank, id) in list.iter().enumerate() {
            // Ranks are small enough for an exact f32.
            #[allow(clippy::cast_precision_loss)]
            let contribution = 1.0 / (k + rank as f32 + 1.0);
            *scores.entry(*id).or_insert(0.0) += contribution;
        }
    }
    let mut fused: Vec<(Uuid, f32)> = scores.into_iter().collect();
    fused.sort_by(|a, b| b.1.total_cmp(&a.1));
    fused
}
