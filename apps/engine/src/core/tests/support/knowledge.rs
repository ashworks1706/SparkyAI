//! Knowledge doubles: a fixed source query registry and an index of canned hits.

use async_trait::async_trait;

use crate::core::traits::knowledge::query::SourceQueries;
use crate::core::traits::knowledge::retrieval::Retriever;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::knowledge::evidence::Evidence;
use crate::core::types::knowledge::query::{
    QueryError, QueryOutcome, QueryRequest, QuerySourceInfo,
};
use crate::core::types::knowledge::retrieval::{RetrievalError, RetrievalQuery};

/// A Retriever double: one canned hit, and the queries it was asked, readable after it moves.
pub struct Stored {
    hits: Vec<Evidence>,
    /// Answers only a query that names no category.
    unfiled: bool,
    asked: std::sync::Arc<std::sync::Mutex<Vec<RetrievalQuery>>>,
}

impl Stored {
    /// An index holding one passage of text under title.
    pub fn holding(title: &str, content: &str) -> Self {
        Self {
            hits: vec![Evidence {
                source_id: uuid::Uuid::nil(),
                chunk_id: uuid::Uuid::nil(),
                key: title.to_owned(),
                title: title.to_owned(),
                content: content.to_owned(),
                url: Some(format!("https://example.test/{title}")),
                fetched_at: chrono::Utc::now() - chrono::Duration::hours(3),
                score: 1.0,
            }],
            unfiled: false,
            asked: std::sync::Arc::default(),
        }
    }

    /// The same index, answering only a query that names no category.
    pub fn unfiled(self) -> Self {
        Self {
            unfiled: true,
            ..self
        }
    }

    /// An index holding nothing.
    pub fn empty() -> Self {
        Self {
            hits: Vec::new(),
            unfiled: false,
            asked: std::sync::Arc::default(),
        }
    }

    /// The queries this double is asked, readable after it moves into a tool.
    pub fn asked(&self) -> std::sync::Arc<std::sync::Mutex<Vec<RetrievalQuery>>> {
        std::sync::Arc::clone(&self.asked)
    }
}

#[async_trait]
impl Retriever for Stored {
    async fn retrieve(
        &self,
        _ctx: &RequestContext,
        query: &RetrievalQuery,
    ) -> Result<Vec<Evidence>, RetrievalError> {
        if let Ok(mut asked) = self.asked.lock() {
            asked.push(query.clone());
        }
        if self.unfiled && query.category.is_some() {
            return Ok(Vec::new());
        }
        Ok(self.hits.clone())
    }
}

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
                fetched_at: None,
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

/// An Embedder double: one fixed vector per text, or a failure, and the texts it was asked.
pub struct FixedEmbedder {
    vector: Option<Vec<f32>>,
    asked: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
}

impl FixedEmbedder {
    /// Embeds every text as vector.
    pub fn returning(vector: Vec<f32>) -> Self {
        Self {
            vector: Some(vector),
            asked: std::sync::Arc::default(),
        }
    }

    /// Fails every call, as an embedding server that is down.
    pub fn failing() -> Self {
        Self {
            vector: None,
            asked: std::sync::Arc::default(),
        }
    }

    /// The texts this double is asked, readable after it moves.
    pub fn asked(&self) -> std::sync::Arc<std::sync::Mutex<Vec<String>>> {
        std::sync::Arc::clone(&self.asked)
    }
}

#[async_trait]
impl crate::core::traits::knowledge::retrieval::Embedder for FixedEmbedder {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, RetrievalError> {
        if let Ok(mut asked) = self.asked.lock() {
            asked.extend(texts.iter().cloned());
        }
        match &self.vector {
            Some(vector) => Ok(texts.iter().map(|_| vector.clone()).collect()),
            None => Err(RetrievalError::Embedding(
                "the embedding server is down".into(),
            )),
        }
    }
}
