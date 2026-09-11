//! Evidence is a retrieved, dated, citable document chunk.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// One retrieved chunk. Citations are built from these, never parsed out of model text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Evidence {
    /// The sources row it came from.
    pub source_id: Uuid,
    /// The chunks row.
    pub chunk_id: Uuid,
    /// Human-readable source name.
    pub title: String,
    /// Chunk text.
    pub content: String,
    /// Canonical page URL, when the source has one.
    pub url: Option<String>,
    /// When the page was fetched.
    pub fetched_at: DateTime<Utc>,
    /// Fused relevance score. Ordering only; not calibrated.
    pub score: f32,
}

impl Evidence {
    /// A citation line for the answer footer.
    pub fn citation(&self) -> String {
        match &self.url {
            Some(url) => format!(
                "{} — {} (fetched {})",
                self.title,
                url,
                self.fetched_at.format("%Y-%m-%d")
            ),
            None => format!(
                "{} (fetched {})",
                self.title,
                self.fetched_at.format("%Y-%m-%d")
            ),
        }
    }

    /// One citation line per source, best first. Chunks sharing a url, or a source id when
    /// there is no url, collapse into the first of them.
    pub fn citations(evidence: &[Evidence]) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        evidence
            .iter()
            .filter(|e| {
                let key = e.url.clone().unwrap_or_else(|| e.source_id.to_string());
                seen.insert(key)
            })
            .map(Evidence::citation)
            .collect()
    }
}
