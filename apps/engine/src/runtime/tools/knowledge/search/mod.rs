//! ReadPublic: two search tools over the ASU sources, one over the stored index and one live.

pub mod campus_map;
pub mod clubs;
pub mod course_catalog;
pub mod courses;
pub mod dining;
pub mod events;
pub mod jobs;
pub mod library_catalog;
pub mod library_hours;
pub mod live;
pub mod news;
pub mod scholarships;
pub mod shuttles;
pub mod social_media;
pub mod sports;
pub mod sports_news;
pub mod stored;
pub mod study_rooms;
pub mod web;

use std::fmt::Write as _;

use chrono::NaiveDate;
use serde_json::{Map, Value, json};

use crate::core::types::knowledge::query::QuerySourceInfo;

/// Name of the tool that searches the stored index.
pub const KNOWLEDGE: &str = "search_knowledge";

/// Name of the tool that fetches a source now.
pub const LIVE: &str = "search_live";

/// Name of the query parameter both tools take.
pub const QUERY: &str = "query";

/// Name of the optional source filter both tools take.
pub const SOURCE: &str = "source";

/// How long the answer of a source stays true.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Freshness {
    /// The answer holds long enough to be worth a stored copy, so the scraper indexes it.
    Stored,
    /// The answer goes stale within minutes, so it reaches the model and nothing else.
    Live,
}

impl Freshness {
    /// Whether the scraper writes an answer of this kind into the retrieval index.
    pub fn indexed(self) -> bool {
        self == Self::Stored
    }
}

/// Every live source the engine offers, in the order the model sees them.
pub fn catalog() -> Vec<Box<dyn LiveSource>> {
    vec![
        Box::new(courses::Courses),
        Box::new(course_catalog::CourseCatalog),
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
        Box::new(dining::Dining),
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

/// One parameter the scraper takes for a source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Param {
    /// Name the scraper takes.
    pub name: &'static str,
    /// What it means.
    pub description: &'static str,
    /// Whether the query is refused without it.
    pub required: bool,
    /// An example value.
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

/// One live ASU source, reached through the source filter of the two search tools.
pub trait LiveSource: Send + Sync {
    /// Registry key the scraper serves it under, and the value of the source filter.
    fn key(&self) -> &'static str;

    /// A few words naming what it answers, listed beside its key in the source filter.
    fn hint(&self) -> &'static str;

    /// Name of the site behind it, as a citation of a live result is labelled.
    fn label(&self) -> &'static str;

    /// The chunks category the scraper writes its pages under.
    fn category(&self) -> &'static str;

    /// The parameters the scraper takes.
    fn params(&self) -> &'static [Param];

    /// The text parameter the query fills. None takes the first text parameter.
    fn query_param(&self) -> Option<&'static str> {
        None
    }

    /// The query as the text parameter takes it, with the words its site cannot match cut.
    fn narrow(&self, query: &str) -> String {
        query.trim().to_owned()
    }

    /// Values for parameters the query cannot carry, applied to what it left empty.
    fn derived(&self, _query: &str, _today: NaiveDate) -> Vec<(&'static str, String)> {
        Vec::new()
    }

    /// How long its answer stays true. Live answers are never indexed.
    fn freshness(&self) -> Freshness {
        Freshness::Stored
    }

    /// Checks filled parameters beyond what each parameter accepts. Err is shown to the model.
    fn check(&self, _params: &Map<String, Value>) -> Result<(), String> {
        Ok(())
    }
}

/// The keys of sources, in catalog order, keeping only those the scraper indexes when stored_only.
pub fn source_keys(sources: &[Box<dyn LiveSource>], stored_only: bool) -> Vec<&'static str> {
    sources
        .iter()
        .filter(|s| !stored_only || s.freshness().indexed())
        .map(|s| s.key())
        .collect()
}

/// The source filter description: lead, then one key and hint per source offered.
pub fn source_help(lead: &str, sources: &[Box<dyn LiveSource>], stored_only: bool) -> String {
    let mut out = lead.trim().to_owned();
    for source in sources {
        if stored_only && !source.freshness().indexed() {
            continue;
        }
        let _ = write!(
            out,
            " {}: {}.",
            source.key(),
            source.hint().trim_end_matches('.')
        );
    }
    out
}

/// The JSON Schema both search tools take: a required query and an optional source filter.
pub fn parameters(query: &str, source: &str, keys: &[&'static str]) -> Value {
    json!({
        "type": "object",
        "properties": {
            QUERY: { "type": "string", "description": query },
            SOURCE: { "type": "string", "enum": keys, "description": source },
        },
        "required": [QUERY],
    })
}

/// The query and source a call carries. Err is shown to the model.
pub fn arguments(
    args: Value,
    keys: &[&'static str],
    tool: &str,
) -> Result<(String, Option<String>), String> {
    let given = match args {
        Value::Object(given) => given,
        Value::Null => Map::new(),
        other => return Err(format!("arguments must be an object, got {other}")),
    };
    if let Some(unknown) = given.keys().find(|k| *k != QUERY && *k != SOURCE) {
        return Err(format!(
            "{tool} has no parameter {unknown}; it takes: {QUERY}, {SOURCE}"
        ));
    }
    let query = match given.get(QUERY) {
        Some(Value::String(text)) => text.trim().to_owned(),
        None | Some(Value::Null) => String::new(),
        Some(other) => return Err(format!("{QUERY} takes text, got {other}")),
    };
    if query.is_empty() {
        return Err(format!(
            "{tool} needs {QUERY}: the words to search for, naming the subject in full"
        ));
    }
    let source = match given.get(SOURCE) {
        None | Some(Value::Null) => None,
        Some(Value::String(text)) if text.trim().is_empty() => None,
        Some(Value::String(text)) => {
            let text = text.trim();
            match keys.iter().find(|k| k.eq_ignore_ascii_case(text)) {
                Some(key) => Some((*key).to_owned()),
                None => {
                    return Err(format!(
                        "{SOURCE} {text:?} is not one of: {}",
                        keys.join(", ")
                    ));
                }
            }
        }
        Some(other) => return Err(format!("{SOURCE} takes text, got {other}")),
    };
    Ok((query, source))
}

/// Whether choice sits in query on its own, rather than inside a longer word.
fn standalone(query: &str, at: usize, len: usize) -> bool {
    let before = query[..at].chars().next_back();
    let after = query[at + len..].chars().next();
    !before.is_some_and(char::is_alphanumeric) && !after.is_some_and(char::is_alphanumeric)
}

/// The choices query names, in declared order. A longer choice wins the words it covers.
pub fn named(query: &str, choices: &[&'static str]) -> Vec<&'static str> {
    let lower = query.to_lowercase();
    let mut longest: Vec<&'static str> = choices.to_vec();
    longest.sort_by_key(|choice| std::cmp::Reverse(choice.len()));
    let mut taken: Vec<(usize, usize)> = Vec::new();
    for choice in longest {
        let needle = choice.to_lowercase();
        let Some(at) = lower
            .match_indices(&needle)
            .map(|(at, _)| at)
            .find(|at| standalone(&lower, *at, needle.len()))
        else {
            continue;
        };
        let span = (at, at + needle.len());
        if taken.iter().any(|(lo, hi)| span.0 < *hi && *lo < span.1) {
            continue;
        }
        taken.push(span);
    }
    choices
        .iter()
        .filter(|choice| {
            let needle = choice.to_lowercase();
            taken
                .iter()
                .any(|(lo, hi)| hi - lo == needle.len() && lower[*lo..*hi] == needle)
        })
        .copied()
        .collect()
}

/// The first YYYY-MM-DD in query.
pub fn dated(query: &str) -> Option<NaiveDate> {
    query
        .split(|c: char| !(c.is_ascii_digit() || c == '-'))
        .find_map(|word| NaiveDate::parse_from_str(word, "%Y-%m-%d").ok())
}

/// The parameters the scraper takes for source, filled from one query. Err names what is missing.
pub fn params_for(
    source: &dyn LiveSource,
    query: &str,
    today: NaiveDate,
) -> Result<Map<String, Value>, String> {
    let params = source.params();
    let fills = source.query_param().or_else(|| {
        params
            .iter()
            .find(|p| p.accepts == Accepts::Text)
            .map(|p| p.name)
    });
    let mut out = Map::new();
    for param in params {
        let value = match param.accepts {
            Accepts::Text if Some(param.name) == fills => source.narrow(query),
            Accepts::OneOf(choices) => named(query, choices)
                .first()
                .map(|choice| (*choice).to_owned())
                .unwrap_or_default(),
            Accepts::AnyOf(choices) => named(query, choices).join(","),
            Accepts::Date => match dated(query) {
                Some(date) => date.format("%Y-%m-%d").to_string(),
                None if param.required => today.format("%Y-%m-%d").to_string(),
                None => String::new(),
            },
            Accepts::Text | Accepts::Flag => String::new(),
        };
        if !value.is_empty() {
            out.insert(param.name.to_owned(), Value::String(value));
        }
    }
    for (name, value) in source.derived(query, today) {
        if !value.is_empty() && !out.contains_key(name) {
            out.insert(name.to_owned(), Value::String(value));
        }
    }
    let missing: Vec<&str> = params
        .iter()
        .filter(|p| p.required && !out.contains_key(p.name))
        .map(|p| p.name)
        .collect();
    if let Some(first) = missing.first() {
        let wanted = params
            .iter()
            .find(|p| p.name == *first)
            .and_then(|p| match p.accepts {
                Accepts::OneOf(choices) | Accepts::AnyOf(choices) => {
                    Some(format!("one of: {}", choices.join(", ")))
                }
                _ => p.example.map(|e| format!("something like {e}")),
            })
            .unwrap_or_else(|| "it".to_owned());
        return Err(format!(
            "{} needs {} in the query; name {wanted}",
            source.key(),
            missing.join(" and ")
        ));
    }
    source.check(&out)?;
    Ok(out)
}

/// Whether source's parameters match what the scraper published for its key. Err names the diff.
pub fn conforms(source: &dyn LiveSource, published: &QuerySourceInfo) -> Result<(), String> {
    let name = source.key();
    if published.indexed != source.freshness().indexed() {
        return Err(format!(
            "{name} is offered as {} but the scraper indexes its answers: {}",
            if source.freshness().indexed() {
                "stored"
            } else {
                "live"
            },
            published.indexed
        ));
    }
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
