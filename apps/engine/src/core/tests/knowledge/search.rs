//! The search tools: one per live source, their schemas, argument checks, boot check vs registry.

use std::sync::Arc;

use serde_json::{Value, json};

use crate::core::tests::support::{FakeQueries, ctx};
use crate::core::traits::tools::Tool;
use crate::core::types::knowledge::query::{QueryParam, QuerySourceInfo};
use crate::core::types::tools::{RiskClass, ToolError};
use crate::runtime::tools::knowledge::search::courses::Courses;
use crate::runtime::tools::knowledge::search::library_hours::LibraryHours;
use crate::runtime::tools::knowledge::search::study_rooms::StudyRooms;
use crate::runtime::tools::knowledge::search::{
    Accepts, LiveSource, Search, arguments, catalog, conforms, definition, tool_name,
};

/// The registry entry the scraper would publish for source.
fn published(source: &dyn LiveSource) -> QuerySourceInfo {
    QuerySourceInfo {
        key: source.key().into(),
        params: source
            .params()
            .iter()
            .map(|p| {
                let (choices, many): (&[&str], bool) = match p.accepts {
                    Accepts::OneOf(c) => (c, false),
                    Accepts::AnyOf(c) => (c, true),
                    Accepts::Flag => (&["true", "false"], false),
                    Accepts::Text | Accepts::Date => (&[], false),
                };
                QueryParam {
                    name: p.name.into(),
                    required: p.required,
                    choices: choices.iter().map(|c| (*c).to_owned()).collect(),
                    many,
                }
            })
            .collect(),
    }
}

fn refused(source: &dyn LiveSource, args: Value) -> String {
    match arguments(source, args) {
        Err(message) => message,
        Ok(params) => unreachable!("expected a refusal, got {params:?}"),
    }
}

#[test]
fn every_source_is_its_own_tool_with_a_unique_name() {
    let sources = catalog();
    let mut names: Vec<String> = sources.iter().map(|s| tool_name(s.key())).collect();
    assert_eq!(names.len(), 15);
    names.sort();
    names.dedup();
    assert_eq!(names.len(), 15, "no two sources share a key");
    for source in &sources {
        let d = definition(source.as_ref(), 90);
        assert!(d.name.starts_with("search_"), "{}", d.name);
        assert!(
            !d.description.trim().is_empty(),
            "{} says what it answers",
            d.name
        );
        assert_eq!(d.risk, RiskClass::ReadPublic);
        assert_eq!(
            d.timeout_secs,
            Some(90),
            "a live fetch outlives the default tool budget"
        );
    }
}

#[test]
fn choices_become_enums_and_a_flag_a_boolean_in_the_schema() {
    let d = definition(&Courses, 90);
    let props = &d.parameters["properties"];
    assert_eq!(d.name, "search_courses");
    assert_eq!(d.parameters["required"], json!(["term"]));
    assert_eq!(
        props["term"]["description"],
        "Term to search: spring, summer or fall and a year. e.g. Fall 2026"
    );
    assert_eq!(props["days"]["type"], "array");
    assert!(
        props["days"]["items"]["enum"]
            .as_array()
            .is_some_and(|e| e.contains(&json!("monday")))
    );
    assert_eq!(props["session"]["enum"], json!(["a", "b", "c", "other"]));
    assert_eq!(props["open_only"]["type"], "boolean");
    let rooms = definition(&StudyRooms, 90);
    assert_eq!(rooms.parameters["properties"]["date"]["format"], "date");
    let hours = definition(&LibraryHours, 90);
    assert_eq!(hours.parameters["properties"], json!({}));
}

#[test]
fn arguments_are_normalized_into_the_strings_the_scraper_takes() {
    let params = arguments(
        &Courses,
        json!({"term": " Fall 2026 ", "days": ["Monday", "wednesday"], "session": "B", "open_only": true, "keywords": null}),
    );
    let Ok(params) = params else {
        unreachable!("valid arguments, got {params:?}")
    };
    assert_eq!(params["term"], "Fall 2026");
    assert_eq!(
        params["days"], "monday,wednesday",
        "choices keep their declared spelling"
    );
    assert_eq!(params["session"], "b");
    assert_eq!(params["open_only"], "true");
    assert!(
        !params.contains_key("keywords"),
        "an absent value is not sent"
    );
    assert!(arguments(&LibraryHours, Value::Null).is_ok_and(|p| p.is_empty()));
}

#[test]
fn a_bad_argument_is_refused_with_what_would_be_accepted() {
    assert!(refused(&Courses, json!({})).contains("needs term"));
    let unknown = refused(&Courses, json!({"term": "Fall 2026", "subject": "CSE"}));
    assert!(
        unknown.contains("no parameter subject") && unknown.contains("keywords"),
        "{unknown}"
    );
    let day = refused(&Courses, json!({"term": "Fall 2026", "days": ["Funday"]}));
    assert!(
        day.contains("\"Funday\" is not one of") && day.contains("monday"),
        "{day}"
    );
    assert!(
        refused(&Courses, json!({"term": "Autumn 26"})).contains("term must look like Fall 2026")
    );
    assert!(
        refused(&Courses, json!({"term": "Fall 2026", "open_only": "maybe"}))
            .contains("true or false")
    );
    let date = refused(&StudyRooms, json!({"library": "hayden", "date": "Sep 14"}));
    assert!(date.contains("2026-09-14"), "{date}");
    assert!(refused(&LibraryHours, json!({"library": "hayden"})).contains("it takes: nothing"));
    assert!(refused(&Courses, json!(["Fall 2026"])).contains("must be an object"));
}

#[test]
fn a_tool_that_matches_what_the_scraper_publishes_conforms() {
    for source in catalog() {
        let served = published(source.as_ref());
        assert_eq!(
            conforms(source.as_ref(), &served),
            Ok(()),
            "{}",
            source.key()
        );
    }
}

#[test]
fn a_tool_that_drifts_from_the_scraper_is_named_at_boot() {
    let mut served = published(&Courses);
    served.params.retain(|p| p.name != "session");
    assert!(conforms(&Courses, &served).is_err_and(|e| e.contains("session")));

    let mut served = published(&Courses);
    served.params.push(QueryParam {
        name: "campus".into(),
        required: false,
        choices: Vec::new(),
        many: false,
    });
    assert!(conforms(&Courses, &served).is_err_and(|e| e.contains("campus")));

    let mut served = published(&Courses);
    if let Some(days) = served.params.iter_mut().find(|p| p.name == "days") {
        days.choices.retain(|c| c != "sunday");
    }
    assert!(conforms(&Courses, &served).is_err_and(|e| e.contains("sunday")));

    let mut served = published(&Courses);
    if let Some(term) = served.params.iter_mut().find(|p| p.name == "term") {
        term.required = false;
    }
    assert!(conforms(&Courses, &served).is_err_and(|e| e.contains("required")));

    let mut served = published(&Courses);
    if let Some(days) = served.params.iter_mut().find(|p| p.name == "days") {
        days.many = false;
    }
    assert!(conforms(&Courses, &served).is_err_and(|e| e.contains("list")));
}

#[tokio::test]
async fn a_call_queues_its_own_source_and_cites_the_page() {
    let queries = FakeQueries::new(vec![published(&Courses)]).answering("courses", "CSE 310 open");
    let sent = queries.sent();
    let tool = Search::new(Box::new(Courses), Arc::new(queries), 90);
    let out = tool
        .call(&ctx(), json!({"term": "Fall 2026", "keywords": "CSE 310"}))
        .await;
    let Ok(output) = out else {
        unreachable!("the source answered, got {out:?}")
    };
    assert!(
        output.content.contains("CSE 310 open"),
        "{}",
        output.content
    );
    assert!(
        output.content.contains("https://example.test/courses"),
        "{}",
        output.content
    );
    assert_eq!(output.sources.len(), 1);
    assert_eq!(
        output.sources[0].url.as_deref(),
        Some("https://example.test/courses")
    );
    let requests = sent.lock().map(|r| r.clone()).unwrap_or_default();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].source, "courses");
    assert_eq!(requests[0].params["keywords"], "CSE 310");
}

#[tokio::test]
async fn a_bad_argument_never_reaches_the_queue() {
    let queries = FakeQueries::new(vec![published(&Courses)]);
    let sent = queries.sent();
    let tool = Search::new(Box::new(Courses), Arc::new(queries), 90);
    let out = tool.call(&ctx(), json!({"keywords": "CSE 310"})).await;
    assert!(
        matches!(out, Err(ToolError::InvalidArguments(_))),
        "{out:?}"
    );
    assert!(sent.lock().is_ok_and(|r| r.is_empty()));
}

#[tokio::test]
async fn a_scraper_refusal_comes_back_as_something_the_model_can_fix() {
    let queries = FakeQueries::new(vec![published(&Courses)])
        .rejecting("courses", "courses returned a page with no readable text");
    let tool = Search::new(Box::new(Courses), Arc::new(queries), 90);
    match tool.call(&ctx(), json!({"term": "Fall 2026"})).await {
        Err(ToolError::InvalidArguments(reason)) => assert!(reason.contains("no readable text")),
        other => unreachable!("expected a correctable refusal, got {other:?}"),
    }
}
