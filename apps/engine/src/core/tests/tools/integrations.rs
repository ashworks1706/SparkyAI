//! The no-auth tools: paper search and Wikipedia, tested on canned JSON bodies.

use crate::runtime::tools::{papers, wiki};

#[test]
fn papers_render_title_authors_and_year() {
    let body = r#"{"data":[
        {"title":"Attention Is All You Need","year":2017,
         "authors":[{"name":"Ashish Vaswani"},{"name":"Noam Shazeer"}],
         "abstract":"A new network architecture, the Transformer.",
         "url":"https://example.org/p1"}]}"#;
    let out = match papers::from_json("transformers", body, 5) {
        Ok(out) => out,
        Err(e) => unreachable!("papers parse: {e}"),
    };
    assert!(
        out.content.contains("Attention Is All You Need"),
        "{}",
        out.content
    );
    assert!(out.content.contains("2017"), "{}", out.content);
    assert!(out.content.contains("Ashish Vaswani"), "{}", out.content);
}

#[test]
fn papers_empty_data_reports_none() {
    let out = match papers::from_json("nothing here", r#"{"data":[]}"#, 5) {
        Ok(out) => out,
        Err(e) => unreachable!("papers parse: {e}"),
    };
    assert!(out.content.contains("No papers"), "{}", out.content);
}

#[test]
fn wikipedia_returns_the_top_page_extract() {
    let body = r#"{"query":{"pages":{
        "736":{"index":1,"title":"Arizona State University","extract":"ASU is a public research university."},
        "999":{"index":2,"title":"Other","extract":"Unrelated."}}}}"#;
    let out = match wiki::from_json("ASU", body, 1500) {
        Ok(out) => out,
        Err(e) => unreachable!("wiki parse: {e}"),
    };
    assert!(
        out.content.starts_with("Arizona State University"),
        "{}",
        out.content
    );
    assert!(
        out.content.contains("public research university"),
        "{}",
        out.content
    );
}

#[test]
fn wikipedia_no_match_says_so() {
    let out = match wiki::from_json("zzzz", r#"{"batchcomplete":""}"#, 1500) {
        Ok(out) => out,
        Err(e) => unreachable!("wiki parse: {e}"),
    };
    assert!(
        out.content.contains("No Wikipedia article"),
        "{}",
        out.content
    );
}
