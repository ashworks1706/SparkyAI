//! The HTTP client every API tool builds, and the short reason a request to it failed.

use std::time::Duration;

/// A client whose every request is bounded by timeout.
pub(crate) fn client(timeout: Duration) -> reqwest::Result<reqwest::Client> {
    reqwest::Client::builder().timeout(timeout).build()
}

/// The kind of failure a request met, without the URL or body it carried.
pub(crate) fn failure(error: &reqwest::Error) -> &'static str {
    if error.is_timeout() {
        "timed out"
    } else if error.is_connect() {
        "could not connect"
    } else {
        "request failed"
    }
}
