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
    /// Key of the sources row it came from.
    pub key: String,
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

/// One source an answer rests on. A client decides how to show it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Citation {
    /// Key of the source behind it: a sources row key, or a live source's registry key.
    pub key: String,
    /// Human-readable source name.
    pub title: String,
    /// Canonical page URL, when the source has one.
    pub url: Option<String>,
}

impl Citation {
    /// The citation as one line, for a client that shows text.
    pub fn line(&self) -> String {
        match &self.url {
            Some(url) => format!("{} - {}", self.title, url),
            None => self.title.clone(),
        }
    }
}

impl Evidence {
    /// The source this chunk credits.
    pub fn citation(&self) -> Citation {
        Citation {
            key: self.key.clone(),
            title: self.title.clone(),
            url: self.url.clone(),
        }
    }

    /// One citation per source, best first. Chunks sharing a url or source id collapse into one.
    pub fn citations(evidence: &[Evidence]) -> Vec<Citation> {
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

/// How long before now a page was fetched, in hours under two days and in days after.
pub fn age(fetched: DateTime<Utc>, now: DateTime<Utc>) -> String {
    let hours = (now - fetched).num_hours();
    match hours {
        ..1 => "under an hour ago".to_owned(),
        1 => "1 hour ago".to_owned(),
        2..48 => format!("{hours} hours ago"),
        _ => format!("{} days ago", hours / 24),
    }
}
