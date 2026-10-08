//! The unit catalog.

use std::collections::HashSet;

use crate::core::types::{Group, Kind};
use crate::units::catalog;

#[test]
fn catalog_ids_are_unique_and_grouped() {
    let units = catalog();
    let ids: HashSet<&str> = units.iter().map(|u| u.id.as_str()).collect();
    assert_eq!(ids.len(), units.len());
    assert!(units.iter().any(|u| u.id == "engine"));
    assert!(units.iter().any(|u| u.id == "eval run"));
    assert!(
        units
            .iter()
            .any(|u| u.id == "prod-up" && u.group == Group::Deploy)
    );
    assert!(
        units
            .iter()
            .all(|u| u.service().is_some() || !u.args.is_empty())
    );
}

#[test]
fn phoenix_is_an_infra_service_behind_its_profile() {
    let units = catalog();
    let phoenix = units.iter().find(|u| u.id == "phoenix");
    assert!(phoenix.is_some_and(|u| u.group == Group::Infra
        && u.url.as_deref() == Some("http://localhost:6006")
        && u.kind
            == Kind::Service {
                service: "phoenix".into(),
                profile: Some("phoenix".into()),
            }));
}
