//! The no-auth tools: paper search, Wikipedia, and Valley Metro, tested on canned JSON bodies.

use crate::runtime::tools::{papers, transit, wiki};

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

#[test]
fn transit_counts_vehicles_by_route() {
    let body = r#"{"entity":[
        {"vehicle":{"trip":{"routeId":"0"}}},
        {"vehicle":{"trip":{"routeId":"0"}}},
        {"vehicle":{"trip":{"route_id":"SME"}}}]}"#;
    let out = match transit::from_json(body, None, 20) {
        Ok(out) => out,
        Err(e) => unreachable!("transit parse: {e}"),
    };
    assert!(
        out.content.contains("3 Valley Metro vehicles"),
        "{}",
        out.content
    );
    assert!(out.content.contains("route 0: 2"), "{}", out.content);
    assert!(out.content.contains("route SME: 1"), "{}", out.content);
}

#[test]
fn transit_resolves_route_ids_to_names() {
    let feed = r#"{"entity":[
        {"vehicle":{"trip":{"routeId":"0"}}},
        {"vehicle":{"trip":{"routeId":"SME"}}}]}"#;
    let routes = "route_id,route_short_name,route_long_name\n\
                  0,0,Central Ave / 1st Ave\n\
                  SME,SME,Streetcar\n";
    let out = match transit::from_json_with_routes(feed, None, 20, routes) {
        Ok(out) => out,
        Err(e) => unreachable!("transit parse: {e}"),
    };
    assert!(
        out.content.contains("Central Ave / 1st Ave (route 0)"),
        "{}",
        out.content
    );
    assert!(
        out.content.contains("Streetcar (route SME)"),
        "{}",
        out.content
    );
}

#[test]
fn transit_filters_by_route_name() {
    let feed = r#"{"entity":[
        {"vehicle":{"trip":{"routeId":"0"}}},
        {"vehicle":{"trip":{"routeId":"SME"}}}]}"#;
    let routes = "route_id,route_short_name,route_long_name\nSME,SME,Streetcar\n";
    let out = match transit::from_json_with_routes(feed, Some("streetcar"), 20, routes) {
        Ok(out) => out,
        Err(e) => unreachable!("transit parse: {e}"),
    };
    assert!(
        out.content.contains("Streetcar (route SME): 1"),
        "{}",
        out.content
    );
    assert!(!out.content.contains("route 0"), "{}", out.content);
}

#[test]
fn transit_filters_to_one_route() {
    let body = r#"{"entity":[
        {"vehicle":{"trip":{"routeId":"0"}}},
        {"vehicle":{"trip":{"routeId":"SME"}}}]}"#;
    let out = match transit::from_json(body, Some("sme"), 20) {
        Ok(out) => out,
        Err(e) => unreachable!("transit parse: {e}"),
    };
    assert!(out.content.contains("route SME: 1"), "{}", out.content);
    assert!(!out.content.contains("route 0"), "{}", out.content);
}
