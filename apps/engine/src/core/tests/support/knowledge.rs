//! Knowledge doubles: a fixed source query registry.

use async_trait::async_trait;

use crate::core::traits::knowledge::query::SourceQueries;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::knowledge::query::{
    QueryError, QueryOutcome, QueryRequest, QuerySourceInfo,
};

/// A SourceQueries double: a fixed registry and a canned outcome per source.
pub struct FakeQueries {
    sources: Vec<QuerySourceInfo>,
    answers: std::collections::HashMap<String, Result<QueryOutcome, String>>,
    sent: std::sync::Arc<std::sync::Mutex<Vec<QueryRequest>>>,
}

impl FakeQueries {
    /// Offers sources and rejects anything not given an answer.
    pub fn new(sources: Vec<QuerySourceInfo>) -> Self {
        Self {
            sources,
            answers: std::collections::HashMap::new(),
            sent: std::sync::Arc::default(),
        }
    }

    /// The requests this double is sent, readable after it moves into a tool.
    pub fn sent(&self) -> std::sync::Arc<std::sync::Mutex<Vec<QueryRequest>>> {
        std::sync::Arc::clone(&self.sent)
    }

    /// Answers source with this page text.
    pub fn answering(mut self, source: &str, text: &str) -> Self {
        self.answers.insert(
            source.to_owned(),
            Ok(QueryOutcome {
                source: source.to_owned(),
                url: format!("https://example.test/{source}"),
                text: text.to_owned(),
            }),
        );
        self
    }

    /// Rejects source with this reason.
    pub fn rejecting(mut self, source: &str, reason: &str) -> Self {
        self.answers
            .insert(source.to_owned(), Err(reason.to_owned()));
        self
    }
}

#[async_trait]
impl SourceQueries for FakeQueries {
    async fn sources(&self) -> Result<Vec<QuerySourceInfo>, QueryError> {
        Ok(self.sources.clone())
    }

    async fn run(
        &self,
        _ctx: &RequestContext,
        request: &QueryRequest,
    ) -> Result<QueryOutcome, QueryError> {
        if let Ok(mut sent) = self.sent.lock() {
            sent.push(request.clone());
        }
        match self.answers.get(&request.source) {
            Some(Ok(outcome)) => Ok(outcome.clone()),
            Some(Err(reason)) => Err(QueryError::Rejected(reason.clone())),
            None => Err(QueryError::Rejected(format!(
                "unknown source {:?}",
                request.source
            ))),
        }
    }
}
