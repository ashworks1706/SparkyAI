//! The classifier that gates profile extraction, and the graph agent that performs it.
//!
//! The classifier runs on every turn and answers one word. The graph agent runs only on the
//! turns it passes, and answers JSON the profile types parse.

use serde::Deserialize;

use std::sync::Arc;
use std::time::Duration;

use crate::agent::harness::task::Task;
use crate::core::traits::detector::FactDetector;
use crate::core::traits::profile::ProfileGraph;
use crate::core::types::context::RequestContext;
use crate::core::types::profile::{ProfileError, ProfileFact};

/// Default instructions for the graph agent.
pub const GRAPH_INSTRUCTIONS: &str = "You extract what a message states about the person who \
wrote it. Answer with one JSON object and nothing else, shaped \
{\"facts\":[{\"subject\":{\"kind\":\"\",\"label\":\"\"},\"relation\":\"\",\
\"object\":{\"kind\":\"\",\"label\":\"\"},\"confidence\":0.0}]}. kind is one of person, course, \
club, place, topic, role. Use the label \"the user\" for the person who wrote the message. \
relation is a short lowercase verb phrase such as studies, prefers, belongs_to, lives_in. \
confidence is between 0 and 1. Extract only what the message states; when it states nothing, \
answer {\"facts\":[]}.";

/// The JSON the graph agent answers with.
#[derive(Debug, Deserialize)]
struct Extraction {
    facts: Vec<ProfileFact>,
}

/// Extracts the facts a turn states, as values the graph stores.
pub struct GraphAgent {
    task: Task,
}

impl GraphAgent {
    /// Builds the graph agent over a task holding the graph agent instructions.
    pub fn new(task: Task) -> Self {
        Self { task }
    }

    /// Extracts the facts turn states.
    ///
    /// # Errors
    /// Returns [`ProfileError::Model`] when the model call fails, and
    /// [`ProfileError::Malformed`] when the answer is not the extraction shape. An answer that
    /// cannot be read is never an empty extraction.
    pub async fn extract(
        &self,
        ctx: &RequestContext,
        turn: &str,
    ) -> Result<Vec<ProfileFact>, ProfileError> {
        let answer = self.task.run(ctx, turn).await?;
        let json = json_object(&answer).ok_or_else(|| {
            ProfileError::Malformed(format!("no JSON object in the answer: {answer}"))
        })?;
        let extraction: Extraction = serde_json::from_str(json)
            .map_err(|e| ProfileError::Malformed(format!("{e}; answer was: {answer}")))?;
        Ok(extraction.facts)
    }
}

/// The outermost JSON object in text, so a fenced or prefaced answer still parses.
fn json_object(text: &str) -> Option<&str> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end < start {
        return None;
    }
    text.get(start..=end)
}

/// The classifier, the graph agent, and the graph they write to.
///
/// Extraction never runs in the request path. The loop hands a finished turn over and returns;
/// what happens after that is detached from the answer the user is waiting for.
pub struct ProfileWriter {
    detector: Arc<dyn FactDetector>,
    agent: GraphAgent,
    graph: Arc<dyn ProfileGraph>,
    budget: Duration,
}

impl ProfileWriter {
    /// Builds the writer over its two prompted sub-agents and the graph.
    pub fn new(
        detector: Arc<dyn FactDetector>,
        agent: GraphAgent,
        graph: Arc<dyn ProfileGraph>,
        budget: Duration,
    ) -> Self {
        Self {
            detector,
            agent,
            graph,
            budget,
        }
    }

    /// Classifies turn, extracts what it states, and writes it. Detached from the request, so
    /// it builds its own context with its own deadline.
    pub async fn record(&self, tenant_id: String, user_id: String, turn: String) {
        // The gate runs on every turn. A model call here would cost one on every greeting.
        if !self.detector.carries_fact(&turn) {
            return;
        }
        let ctx = RequestContext::new(tenant_id, user_id, self.budget);
        let facts = match self.agent.extract(&ctx, &turn).await {
            Ok(facts) => facts,
            Err(error) => {
                tracing::warn!(error = %error, "profile extraction failed");
                return;
            }
        };
        if facts.is_empty() {
            return;
        }
        match self.graph.upsert(&ctx, &facts).await {
            Ok(()) => tracing::info!(facts = facts.len(), "profile graph updated"),
            Err(error) => tracing::warn!(error = %error, "profile graph write failed"),
        }
    }
}
