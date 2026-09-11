//! The default guardrail. Denied phrases, a length ceiling, and an empty answer check.
//!
//! Every rule is configuration. A deployment that needs a model-backed check implements the
//! trait instead; nothing here is the only possible gate.

use async_trait::async_trait;

use crate::core::traits::guardrail::Guardrail;
use crate::core::types::context::RequestContext;
use crate::core::types::guardrail::{Stage, Verdict};

/// What the default guardrail refuses.
#[derive(Debug, Clone)]
pub struct Rules {
    /// Lowercase phrases that block a response wherever they appear.
    pub denied_phrases: Vec<String>,
    /// Longest answer allowed. Zero removes the limit.
    pub max_answer_chars: usize,
    /// Text shown in place of a blocked response.
    pub replacement: String,
}

impl Default for Rules {
    fn default() -> Self {
        Self::from(&crate::core::config::Guardrail::default())
    }
}

impl From<&crate::core::config::Guardrail> for Rules {
    fn from(cfg: &crate::core::config::Guardrail) -> Self {
        Self {
            denied_phrases: cfg
                .denied_phrases
                .iter()
                .map(|p| p.to_lowercase())
                .collect(),
            max_answer_chars: cfg.max_answer_chars,
            replacement: cfg.replacement.clone(),
        }
    }
}

/// Checks a response against a fixed set of rules.
#[derive(Debug, Clone, Default)]
pub struct RuleGuardrail {
    rules: Rules,
}

impl RuleGuardrail {
    /// Builds the guardrail over its rules.
    pub fn new(rules: Rules) -> Self {
        Self { rules }
    }

    fn block(&self, reason: impl Into<String>) -> Verdict {
        Verdict::Block {
            replacement: self.rules.replacement.clone(),
            reason: reason.into(),
        }
    }
}

#[async_trait]
impl Guardrail for RuleGuardrail {
    async fn check(&self, _ctx: &RequestContext, stage: Stage, text: &str) -> Verdict {
        let lowered = text.to_lowercase();
        if let Some(phrase) = self
            .rules
            .denied_phrases
            .iter()
            .find(|p| !p.is_empty() && lowered.contains(p.as_str()))
        {
            return self.block(format!("denied phrase {phrase}"));
        }
        // An answer is what the user reads. A capability branch carries tool calls and often no
        // text at all, so neither length nor emptiness is checked there.
        if stage == Stage::Answer {
            if text.trim().is_empty() {
                return self.block("the answer was empty");
            }
            if self.rules.max_answer_chars > 0 && text.chars().count() > self.rules.max_answer_chars
            {
                return self.block(format!(
                    "the answer ran to {} characters",
                    text.chars().count()
                ));
            }
        }
        Verdict::Pass
    }
}
