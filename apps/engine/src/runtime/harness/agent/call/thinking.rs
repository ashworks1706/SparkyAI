//! Whether one model call of the loop thinks.

use crate::core::types::agent::thinking::{
    ThinkingChoice, ThinkingMode, ThinkingReason, ThinkingRules, words,
};

/// What the loop knows about a step before it calls the model.
#[derive(Debug, Clone, Copy)]
pub struct StepSignals<'a> {
    /// The call is offered no tools.
    pub answer_only: bool,
    /// The turns of this request hold a tool result.
    pub tool_results: bool,
    /// The question of this request. Empty when resuming after an approval.
    pub input: &'a str,
    /// Evidence chunks retrieval found for the question.
    pub evidence: usize,
}

/// Decides thinking for one call. The first rule that applies wins.
pub fn decide(rules: &ThinkingRules, step: &StepSignals<'_>) -> ThinkingChoice {
    let choose = |on, reason| ThinkingChoice { on, reason };
    match rules.mode {
        ThinkingMode::On => return choose(true, ThinkingReason::Mode),
        ThinkingMode::Off => return choose(false, ThinkingReason::Mode),
        ThinkingMode::Auto => {}
    }
    if step.answer_only {
        choose(false, ThinkingReason::AnswerOnly)
    } else if step.tool_results {
        choose(rules.after_tools, ThinkingReason::AfterTools)
    } else if has_cue(step.input, &rules.cues) {
        choose(true, ThinkingReason::Cue)
    } else if step.input.chars().count() > rules.max_quick_chars {
        choose(true, ThinkingReason::Long)
    } else if step.evidence == 0 {
        choose(true, ThinkingReason::NoEvidence)
    } else {
        choose(false, ThinkingReason::Quick)
    }
}

/// Whether input holds any cue as whole words, ignoring case.
fn has_cue(input: &str, cues: &[String]) -> bool {
    let asked = words(input);
    cues.iter().any(|cue| {
        let cue = words(cue);
        !cue.is_empty()
            && asked
                .windows(cue.len())
                .any(|window| window == cue.as_slice())
    })
}
