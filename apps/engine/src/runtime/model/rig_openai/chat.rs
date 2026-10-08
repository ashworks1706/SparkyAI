//! RigChat: the ModelProvider over the chat-completions model of Rig.

use ::rig_core::completion::{
    AssistantContent, CompletionModel as _, CompletionRequest, FinishReason as RigFinish,
    ToolDefinition as RigTool,
};
use ::rig_core::message::ToolChoice;
use ::rig_core::providers::openai::{CompletionModel, CompletionsClient};
use ::rig_core::streaming::StreamedAssistantContent;
use async_trait::async_trait;
use futures::StreamExt as _;
use tokio::sync::mpsc::UnboundedSender;

use crate::core::traits::model::ModelProvider;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::model::{ModelDelta, ModelError, ModelRequest, ModelResponse, Usage};
use crate::runtime::model::rig_openai::convert::{
    finish_reason, from_rig, map_error, to_rig, tool_to_rig, with_thinking,
};

/// ModelProvider over the chat-completions model of Rig.
#[derive(Clone)]
pub struct RigChat {
    model: CompletionModel,
    name: String,
    /// Provider-specific request fields sent with every completion.
    additional_params: serde_json::Value,
}

impl RigChat {
    /// Wraps model_name on client. additional_params is sent verbatim with every completion.
    pub fn new(
        client: CompletionsClient,
        model_name: impl Into<String>,
        additional_params: serde_json::Value,
    ) -> Self {
        let name = model_name.into();
        Self {
            model: CompletionModel::new(client, name.clone()),
            name,
            additional_params,
        }
    }
}

impl RigChat {
    /// The Rig request for req.
    fn request(&self, req: &ModelRequest) -> Result<CompletionRequest, ModelError> {
        let (preamble, chat_history) = to_rig(&req.messages)?;
        let tools: Vec<RigTool> = req.tools.iter().map(tool_to_rig).collect();
        Ok(CompletionRequest {
            model: None,
            preamble,
            chat_history,
            documents: Vec::new(),
            tool_choice: (!tools.is_empty()).then_some(ToolChoice::Auto),
            tools,
            temperature: Some(f64::from(req.temperature)),
            max_tokens: Some(u64::from(req.max_tokens)),
            // llama-server honours chat_template_kwargs and the llama.cpp sampling fields.
            additional_params: Some(with_thinking(&self.additional_params, req.thinking)),
            output_schema: None,
            record_telemetry_content: false,
        })
    }

    /// The core response for a Rig choice, its usage, and the finish reason the provider gave.
    fn response(
        &self,
        choice: Vec<AssistantContent>,
        usage: ::rig_core::completion::Usage,
        reported: Option<&RigFinish>,
    ) -> ModelResponse {
        let (content, reasoning, tool_calls) = from_rig(choice);
        let finish_reason = finish_reason(reported, &content, &tool_calls);
        ModelResponse {
            content,
            reasoning,
            tool_calls,
            finish_reason,
            usage: Usage {
                prompt_tokens: u32::try_from(usage.input_tokens).unwrap_or(u32::MAX),
                completion_tokens: u32::try_from(usage.output_tokens).unwrap_or(u32::MAX),
            },
            model: self.name.clone(),
        }
    }
}

#[async_trait]
impl ModelProvider for RigChat {
    async fn generate(
        &self,
        _ctx: &RequestContext,
        req: ModelRequest,
    ) -> Result<ModelResponse, ModelError> {
        let request = self.request(&req)?;
        let response = self.model.completion(request).await.map_err(map_error)?;
        Ok(self.response(response.choice, response.usage, None))
    }

    async fn stream(
        &self,
        _ctx: &RequestContext,
        req: ModelRequest,
        deltas: UnboundedSender<ModelDelta>,
    ) -> Result<ModelResponse, ModelError> {
        let request = self.request(&req)?;
        let mut stream = self.model.stream(request).await.map_err(map_error)?;
        let mut usage = ::rig_core::completion::Usage::new();
        let mut reported = None;
        let mut finished = false;
        while let Some(item) = stream.next().await {
            // A closed receiver means nobody is watching; the completion continues.
            match item.map_err(map_error)? {
                StreamedAssistantContent::ReasoningDelta { reasoning, .. } => {
                    let _ = deltas.send(ModelDelta::Reasoning(reasoning));
                }
                StreamedAssistantContent::Text(text) => {
                    let _ = deltas.send(ModelDelta::Text(text.text));
                }
                StreamedAssistantContent::Final(last) => {
                    usage = last.usage;
                    reported = last.finish_reason;
                    finished = true;
                }
                _ => {}
            }
        }
        if !finished {
            return Err(ModelError::Transport(
                "the model stream ended before the completion finished".into(),
            ));
        }
        let choice = std::mem::take(&mut stream.choice);
        Ok(self.response(choice, usage, reported.as_ref()))
    }
}
