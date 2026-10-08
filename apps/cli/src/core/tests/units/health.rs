//! Health probes and port checks.

use std::time::Duration;

use crate::units::health::{address_of, served};

#[test]
fn a_unit_url_reduces_to_the_address_a_port_check_can_reach() {
    assert_eq!(
        address_of("http://localhost:8080").as_deref(),
        Some("localhost:8080")
    );
    assert_eq!(
        address_of("localhost:5432").as_deref(),
        Some("localhost:5432")
    );
    assert_eq!(
        address_of("http://localhost:8000/v1").as_deref(),
        Some("localhost:8000")
    );
    // A unit bound to every interface is reached on loopback from here.
    assert_eq!(
        address_of("http://0.0.0.0:8080").as_deref(),
        Some("127.0.0.1:8080")
    );
    assert_eq!(address_of("http://localhost").as_deref(), None);
    assert_eq!(address_of("").as_deref(), None);
}

#[test]
fn a_port_already_listening_reads_as_served() {
    let Ok(listener) = std::net::TcpListener::bind("127.0.0.1:0") else {
        unreachable!("a loopback port was free")
    };
    let Ok(addr) = listener.local_addr() else {
        unreachable!("a bound listener has an address")
    };
    let timeout = Duration::from_millis(500);
    assert!(served(&addr.to_string(), timeout), "{addr} is bound");

    drop(listener);
    assert!(
        !served(&format!("127.0.0.1:{}", addr.port()), timeout),
        "nothing listens once it is closed"
    );
}
