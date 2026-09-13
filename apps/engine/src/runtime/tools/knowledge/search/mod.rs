//! ReadPublic: one search tool per live ASU source. A call checks arguments, queues a scraper job.

pub mod campus_map;
pub mod clubs;
pub mod courses;
pub mod events;
pub mod jobs;
pub mod library_catalog;
pub mod library_hours;
pub mod news;
pub mod scholarships;
pub mod shuttles;
pub mod social_media;
pub mod sports;
pub mod sports_news;
pub mod study_rooms;
pub mod web;

use std::fmt::Write as _;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Map, Value, json};

use crate::core::traits::knowledge::query::SourceQueries;
use crate::core::traits::tools::Tool;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::knowledge::evidence::Citation;
use crate::core::types::knowledge::query::{QueryError, QueryRequest, QuerySourceInfo};
use crate::core::types::tools::{RiskClass, ToolDefinition, ToolError, ToolOutput};
use crate::runtime::tools::structured;

/// Prefix of every search tool name. The rest is the source key.
pub const PREFIX: &str = "search_";

/// The tool name a source is offered under.
pub fn tool_name(key: &str) -> String {
    format!("{PREFIX}{key}")
}

/// Every live source the engine offers, in the order the model sees them.
pub fn catalog() -> Vec<Box<dyn LiveSource>> {
    vec![
        Box::new(courses::Courses),
        Box::new(scholarships::Scholarships),
        Box::new(events::Events),
        Box::new(clubs::Clubs),
        Box::new(news::News),
        Box::new(library_catalog::LibraryCatalog),
        Box::new(library_hours::LibraryHours),
        Box::new(study_rooms::StudyRooms),
        Box::new(sports::Sports),
        Box::new(sports_news::SportsNews),
        Box::new(shuttles::Shuttles),
        Box::new(campus_map::CampusMap),
        Box::new(social_media::SocialMedia),
        Box::new(jobs::Jobs),
        Box::new(web::Web),
    ]
}

/// What a parameter accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Accepts {
    /// Any text.
    Text,
    /// One of these values.
    OneOf(&'static [&'static str]),
    /// Any number of these values.
    AnyOf(&'static [&'static str]),
    /// A calendar date, YYYY-MM-DD.
    Date,
    /// true or false.
    Flag,
}

/// One parameter a search tool takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Param {
    /// Name the model passes.
    pub name: &'static str,
    /// What it means, for the model.
    pub description: &'static str,
    /// Whether the call is refused without it.
    pub required: bool,
    /// An example value shown to the model.
    pub example: Option<&'static str>,
    /// What values it accepts.
    pub accepts: Accepts,
}

impl Param {
    /// An optional text parameter.
    pub const fn text(name: &'static str, description: &'static str) -> Self {
        Self {
            name,
            description,
            required: false,
            example: None,
            accepts: Accepts::Text,
        }
    }

    /// An optional parameter that takes one of choices.
    pub const fn one_of(
        name: &'static str,
        description: &'static str,
        choices: &'static [&'static str],
    ) -> Self {
        Self {
            accepts: Accepts::OneOf(choices),
            ..Self::text(name, description)
        }
    }

    /// An optional parameter that takes any number of choices.
    pub const fn any_of(
        name: &'static str,
        description: &'static str,
        choices: &'static [&'static str],
    ) -> Self {
        Self {
            accepts: Accepts::AnyOf(choices),
            ..Self::text(name, description)
        }
    }

    /// An optional date parameter.
    pub const fn date(name: &'static str, description: &'static str) -> Self {
        Self {
            accepts: Accepts::Date,
            ..Self::text(name, description)
        }
    }

    /// An optional true or false parameter.
    pub const fn flag(name: &'static str, description: &'static str) -> Self {
        Self {
            accepts: Accepts::Flag,
            ..Self::text(name, description)
        }
    }

    /// The same parameter, required.
    pub const fn required(self) -> Self {
        Self {
            required: true,
            ..self
        }
    }

    /// The same parameter, with an example value.
    pub const fn example(self, example: &'static str) -> Self {
        Self {
            example: Some(example),
            ..self
        }
    }

    /// The JSON Schema of the parameter.
    fn schema(&self) -> Value {
        let mut description = self.description.to_owned();
        if let Some(example) = self.example {
            let _ = write!(description, " e.g. {example}");
        }
        match self.accepts {
            Accepts::Text => json!({ "type": "string", "description": description }),
            Accepts::OneOf(choices) => {
                json!({ "type": "string", "enum": choices, "description": description })
            }
            Accepts::AnyOf(choices) => json!({
                "type": "array",
                "items": { "type": "string", "enum": choices },
                "description": description
            }),
            Accepts::Date => {
                json!({ "type": "string", "format": "date", "description": description })
            }
            Accepts::Flag => json!({ "type": "boolean", "description": description }),
        }
    }

    /// The argument as the string the scraper takes. Err names what was wrong with it.
    fn normalize(&self, value: &Value) -> Result<String, String> {
        let text = match value {
            Value::String(text) => text.trim().to_owned(),
            Value::Bool(flag) => flag.to_string(),
            Value::Number(number) => number.to_string(),
            Value::Array(items) => items
                .iter()
                .map(|item| match item {
                    Value::String(text) => Ok(text.trim().to_owned()),
                    other => Err(format!("{} takes text values, got {other}", self.name)),
                })
                .collect::<Result<Vec<_>, _>>()?
                .join(","),
            other => return Err(format!("{} takes text, got {other}", self.name)),
        };
        match self.accepts {
            Accepts::Text => Ok(text),
            Accepts::OneOf(choices) => match pick(&text, choices) {
                Some(choice) => Ok(choice.to_owned()),
                None => Err(not_one_of(self.name, &text, choices)),
            },
            Accepts::AnyOf(choices) => text
                .split(',')
                .map(str::trim)
                .filter(|part| !part.is_empty())
                .map(|part| pick(part, choices).ok_or_else(|| not_one_of(self.name, part, choices)))
                .collect::<Result<Vec<_>, _>>()
                .map(|picked| picked.join(",")),
            Accepts::Date => chrono::NaiveDate::parse_from_str(&text, "%Y-%m-%d")
                .map(|date| date.format("%Y-%m-%d").to_string())
                .map_err(|_| format!("{} must be a date like 2026-09-14, got {text:?}", self.name)),
            Accepts::Flag => match text.to_ascii_lowercase().as_str() {
                "true" => Ok("true".to_owned()),
                "false" => Ok("false".to_owned()),
                _ => Err(format!("{} must be true or false, got {text:?}", self.name)),
            },
        }
    }

    /// The values the scraper has to accept for this parameter, and whether it takes a list.
    fn values(&self) -> Option<(&'static [&'static str], bool)> {
        match self.accepts {
            Accepts::OneOf(choices) => Some((choices, false)),
            Accepts::AnyOf(choices) => Some((choices, true)),
            Accepts::Flag => Some((&["true", "false"], false)),
            Accepts::Text | Accepts::Date => None,
        }
    }
}

/// The choice text names, compared without case.
fn pick<'a>(text: &str, choices: &[&'a str]) -> Option<&'a str> {
    choices
        .iter()
        .find(|choice| choice.eq_ignore_ascii_case(text))
        .copied()
}

/// The refusal for a value that is not one of choices.
fn not_one_of(name: &str, value: &str, choices: &[&str]) -> String {
    format!("{name} {value:?} is not one of: {}", choices.join(", "))
}

/// One live ASU source the model may search.
pub trait LiveSource: Send + Sync {
    /// Registry key the scraper serves it under.
    fn key(&self) -> &'static str;

    /// What it answers, for the model.
    fn description(&self) -> &'static str;

    /// The parameters it takes.
    fn params(&self) -> &'static [Param];

    /// Checks normalized arguments beyond what each parameter accepts. Err is shown to the model.
    fn check(&self, _params: &Map<String, Value>) -> Result<(), String> {
        Ok(())
    }
}

/// The tool definition of a source.
pub fn definition(source: &dyn LiveSource, timeout_secs: u64) -> ToolDefinition {
    let mut properties = Map::new();
    let mut required = Vec::new();
    for param in source.params() {
        properties.insert(param.name.to_owned(), param.schema());
        if param.required {
            required.push(param.name);
        }
    }
    ToolDefinition {
        name: tool_name(source.key()),
        description: source.description().to_owned(),
        parameters: json!({ "type": "object", "properties": properties, "required": required }),
        risk: RiskClass::ReadPublic,
        sequential: false,
        timeout_secs: Some(timeout_secs),
    }
}

/// The arguments of a call as the parameters the scraper takes. Err is shown to the model.
pub fn arguments(source: &dyn LiveSource, args: Value) -> Result<Map<String, Value>, String> {
    let given = match args {
        Value::Object(given) => given,
        Value::Null => Map::new(),
        other => return Err(format!("arguments must be an object, got {other}")),
    };
    let params = source.params();
    let taken = || {
        let names: Vec<&str> = params.iter().map(|p| p.name).collect();
        if names.is_empty() {
            "nothing".to_owned()
        } else {
            names.join(", ")
        }
    };
    if let Some(unknown) = given.keys().find(|k| !params.iter().any(|p| p.name == *k)) {
        return Err(format!(
            "{} has no parameter {unknown}; it takes: {}",
            tool_name(source.key()),
            taken()
        ));
    }
    let mut out = Map::new();
    for param in params {
        let value = match given.get(param.name) {
            None | Some(Value::Null) => String::new(),
            Some(value) => param.normalize(value)?,
        };
        if value.is_empty() {
            if param.required {
                return Err(format!("{} needs {}", tool_name(source.key()), param.name));
            }
            continue;
        }
        out.insert(param.name.to_owned(), Value::String(value));
    }
    source.check(&out)?;
    Ok(out)
}

/// Whether source's parameters match what the scraper published for its key. Err names the diff.
pub fn conforms(source: &dyn LiveSource, published: &QuerySourceInfo) -> Result<(), String> {
    let name = tool_name(source.key());
    for param in source.params() {
        let Some(served) = published.params.iter().find(|p| p.name == param.name) else {
            return Err(format!(
                "{name} takes {} but the scraper does not",
                param.name
            ));
        };
        if served.required != param.required {
            return Err(format!(
                "{name} and the scraper disagree on whether {} is required",
                param.name
            ));
        }
        if let Some((values, many)) = param.values() {
            if served.many != many {
                return Err(format!(
                    "{name} and the scraper disagree on whether {} takes a list",
                    param.name
                ));
            }
            if let Some(missing) = values
                .iter()
                .find(|v| !served.choices.iter().any(|c| c.eq_ignore_ascii_case(v)))
            {
                return Err(format!(
                    "{name} offers {missing:?} for {} but the scraper does not accept it",
                    param.name
                ));
            }
        }
    }
    if let Some(extra) = published
        .params
        .iter()
        .find(|served| !source.params().iter().any(|p| p.name == served.name))
    {
        return Err(format!(
            "the scraper takes {} for {} but {name} does not offer it",
            extra.name, published.key
        ));
    }
    Ok(())
}

/// A search tool over one live source.
pub struct Search {
    source: Box<dyn LiveSource>,
    queries: Arc<dyn SourceQueries>,
    definition: ToolDefinition,
}

impl Search {
    /// Builds the tool for source. timeout_secs overrides the default of the agent.
    pub fn new(
        source: Box<dyn LiveSource>,
        queries: Arc<dyn SourceQueries>,
        timeout_secs: u64,
    ) -> Self {
        let definition = definition(source.as_ref(), timeout_secs);
        Self {
            source,
            queries,
            definition,
        }
    }
}

#[async_trait]
impl Tool for Search {
    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    async fn call(&self, ctx: &RequestContext, args: Value) -> Result<ToolOutput, ToolError> {
        let params = arguments(self.source.as_ref(), args).map_err(ToolError::InvalidArguments)?;
        let request = QueryRequest {
            source: self.source.key().to_owned(),
            params,
        };
        let outcome = self.queries.run(ctx, &request).await.map_err(|e| match e {
            // A rejection reaches the model as an argument error.
            QueryError::Rejected(reason) => ToolError::InvalidArguments(reason),
            QueryError::Cancelled => ToolError::Cancelled,
            QueryError::Timeout(_) => ToolError::Timeout,
            absent @ QueryError::NoWorker(_) => ToolError::Failed(absent.to_string()),
            store @ QueryError::Store(_) => ToolError::Failed(store.to_string()),
        })?;
        Ok(ToolOutput {
            content: format!(
                "Live result from {} ({}):\n\n{}",
                outcome.source, outcome.url, outcome.text
            ),
            data: structured(&outcome),
            sources: vec![Citation {
                title: outcome.source.clone(),
                url: Some(outcome.url.clone()),
            }],
        })
    }
}
