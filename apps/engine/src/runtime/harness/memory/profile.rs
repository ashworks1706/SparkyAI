//! The classifier that gates profile extraction, and the graph agent that performs it.
//!
//! The classifier runs on every turn and answers one word. The graph agent runs only on the
//! turns it passes, and answers JSON the profile types parse.

use serde::Deserialize;

use std::fmt::Write as _;
use std::sync::Arc;
use std::time::Duration;

use crate::core::traits::memory::detector::FactDetector;
use crate::core::traits::memory::profile::ProfileGraph;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::memory::profile::{ProfileError, ProfileFact, ProfileRelation};
use crate::runtime::harness::agent::task::Task;

/// Default instructions for the graph agent.
pub const GRAPH_INSTRUCTIONS: &str = "You extract what a message states about the person who \
wrote it. Answer with one JSON object and nothing else, shaped \
{\"facts\":[{\"subject\":{\"kind\":\"\",\"label\":\"\"},\"relation\":\"\",\
\"object\":{\"kind\":\"\",\"label\":\"\"},\"confidence\":0.0}]}. kind is one of person, \
course, club, place, topic, role. Use the label \"the user\" for the person who wrote the \
message. confidence is between 0 and 1.\n\nrelation must be one of: studies, prefers, \
belongs_to, lives_in, works_at, enrolled_in, named, interested_in. Pick the closest one and \
never invent another.\n\nRecord the state a message leaves the person in, not the event that \
changed it. I switched my major to physics is studies physics, not switched physics. I dropped \
CSE 110 states nothing to record. Writing the event instead of the state hides that the new \
fact replaces an old one.\n\nExtract only what the message states; when it states nothing, \
answer {\"facts\":[]}.\n\nExamples.\nI am a computer science major at ASU. -> \
{\"facts\":[{\"subject\":{\"kind\":\"person\",\"label\":\"the user\"},\
\"relation\":\"studies\",\"object\":{\"kind\":\"topic\",\"label\":\"computer \
science\"},\"confidence\":1.0},{\"subject\":{\"kind\":\"person\",\"label\":\"the \
user\"},\"relation\":\"enrolled_in\",\"object\":{\"kind\":\"place\",\"label\":\
\"ASU\"},\"confidence\":1.0}]}\nI switched my major to physics. -> \
{\"facts\":[{\"subject\":{\"kind\":\"person\",\"label\":\"the user\"},\
\"relation\":\"studies\",\"object\":{\"kind\":\"topic\",\"label\":\"physics\"},\
\"confidence\":1.0}]}\nMy name is Alex. -> {\"facts\":[{\"subject\":{\"kind\":\
\"person\",\"label\":\"the user\"},\"relation\":\"named\",\"object\":{\"kind\":\
\"person\",\"label\":\"Alex\"},\"confidence\":1.0}]}";

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
    /// Returns [ProfileError::Model] when the model call fails, and
    /// [ProfileError::Malformed] when the answer is not the extraction shape. An answer that
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
    reconciler: Option<Reconciler>,
    graph: Arc<dyn ProfileGraph>,
    budget: Duration,
    min_confidence: f32,
}

/// Whether an extracted fact names both entities and the relation, at or above floor.
pub fn keepable(fact: &ProfileFact, floor: f32) -> bool {
    !fact.subject.label.trim().is_empty()
        && !fact.object.label.trim().is_empty()
        && !fact.relation.trim().is_empty()
        && fact.confidence >= floor
}

impl ProfileWriter {
    /// Builds the writer over its two prompted sub-agents and the graph. Facts below
    /// min_confidence are dropped before they are written.
    pub fn new(
        detector: Arc<dyn FactDetector>,
        agent: GraphAgent,
        reconciler: Option<Reconciler>,
        graph: Arc<dyn ProfileGraph>,
        budget: Duration,
        min_confidence: f32,
    ) -> Self {
        Self {
            detector,
            agent,
            reconciler,
            graph,
            budget,
            min_confidence,
        }
    }

    /// Withdraws the statements a new fact makes false, before the new one is written.
    ///
    /// A reconciler that cannot be reached leaves what is recorded alone. Keeping a stale fact
    /// is recoverable; removing a true one is not.
    async fn reconcile(&self, ctx: &RequestContext, facts: &[ProfileFact]) {
        let Some(reconciler) = &self.reconciler else {
            return;
        };
        for fact in facts {
            let existing = match self
                .graph
                .matching(ctx, &fact.subject.label, &fact.relation)
                .await
            {
                Ok(existing) => existing,
                Err(error) => {
                    tracing::warn!(error = %error, "could not read what is already recorded");
                    continue;
                }
            };
            // A statement the person just repeated is already there and replaces nothing.
            let candidates: Vec<ProfileRelation> = existing
                .into_iter()
                .filter(|e| !e.object.label.eq_ignore_ascii_case(&fact.object.label))
                .collect();
            let superseded = match reconciler.superseded(ctx, fact, &candidates).await {
                Ok(superseded) => superseded,
                Err(error) => {
                    tracing::warn!(error = %error, "reconciler failed; nothing is withdrawn");
                    continue;
                }
            };
            for index in superseded {
                let Some(stale) = candidates.get(index) else {
                    continue;
                };
                match self.graph.drop_relation(ctx, stale).await {
                    Ok(true) => tracing::info!(statement = %stale, "withdrew a superseded fact"),
                    Ok(false) => {}
                    Err(error) => tracing::warn!(error = %error, "could not withdraw a fact"),
                }
            }
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
        let facts: Vec<ProfileFact> = match self.agent.extract(&ctx, &turn).await {
            Ok(facts) => facts
                .into_iter()
                .filter(|f| keepable(f, self.min_confidence))
                .collect(),
            Err(error) => {
                tracing::warn!(error = %error, "profile extraction failed");
                return;
            }
        };
        if facts.is_empty() {
            return;
        }
        self.reconcile(&ctx, &facts).await;
        match self.graph.upsert(&ctx, &facts).await {
            Ok(()) => tracing::info!(facts = facts.len(), "profile graph updated"),
            Err(error) => tracing::warn!(error = %error, "profile graph write failed"),
        }
    }
}

/// Default instructions for the reconciler.
///
/// Scoped statements are the trap. mem0 reports that a fact extractor which strips scope turns
/// two compatible preferences into a contradiction, so the scope stays in the text the model
/// reads and the instructions name the case.
pub const RECONCILE_INSTRUCTIONS: &str = "A person stated something new. Some of what is \
already recorded about them may now be wrong. Decide which recorded statements the new one \
replaces. A statement is replaced only when the new one makes it false: a changed major, a \
dropped class, a moved home. Two statements that can both be true are both kept, including \
preferences that apply in different situations and facts that simply accumulate. Answer with \
the numbers of the statements to remove, separated by commas, or the word none. Write no \
other word.";

/// Decides which recorded statements a new one replaces.
pub struct Reconciler {
    task: Task,
}

impl Reconciler {
    /// Builds the reconciler over a task holding its instructions.
    pub fn new(task: Task) -> Self {
        Self { task }
    }

    /// The statements the new fact replaces, by index into existing.
    ///
    /// # Errors
    /// Returns [ProfileError::Model] when the call fails.
    pub async fn superseded(
        &self,
        ctx: &RequestContext,
        fact: &ProfileFact,
        existing: &[ProfileRelation],
    ) -> Result<Vec<usize>, ProfileError> {
        if existing.is_empty() {
            return Ok(Vec::new());
        }
        let mut prompt = format!(
            "New statement: {} {} {}\n\nAlready recorded:\n",
            fact.subject.label, fact.relation, fact.object.label
        );
        for (i, e) in existing.iter().enumerate() {
            let _ = writeln!(prompt, "{}. {}", i + 1, e);
        }
        let answer = self.task.run(ctx, &prompt).await?;
        Ok(parse_indices(&answer, existing.len()))
    }
}

/// Reads the reconciler answer as indices into the list it was shown.
///
/// Anything unreadable is read as none. Removing a statement on a misparse is the one outcome
/// that loses what a person said.
pub fn parse_indices(answer: &str, len: usize) -> Vec<usize> {
    let lowered = answer.trim().to_ascii_lowercase();
    if lowered.contains("none") {
        return Vec::new();
    }
    let mut out: Vec<usize> = lowered
        .split(|c: char| !c.is_ascii_digit())
        .filter(|piece| !piece.is_empty())
        .filter_map(|piece| piece.parse::<usize>().ok())
        .filter(|n| *n >= 1 && *n <= len)
        .map(|n| n - 1)
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}
