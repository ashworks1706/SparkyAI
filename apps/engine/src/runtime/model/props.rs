//! What a llama-server reports about itself: the context one slot holds.

use std::time::Duration;

use secrecy::{ExposeSecret, SecretString};

/// How long the engine waits for the server to answer.
const TIMEOUT: Duration = Duration::from_secs(5);

/// The tokens one slot of the llama-server at base_url holds, prompt and completion together.
///
/// # Errors
/// Returns the reason when the server does not answer or reports no context size.
pub async fn slot_context(base_url: &str, api_key: &SecretString) -> Result<u32, String> {
    let root = base_url.trim_end_matches('/').trim_end_matches("/v1");
    let client = reqwest::Client::builder()
        .timeout(TIMEOUT)
        .build()
        .map_err(|e| e.to_string())?;
    let mut request = client.get(format!("{root}/props"));
    if !api_key.expose_secret().is_empty() {
        request = request.bearer_auth(api_key.expose_secret());
    }
    let props: serde_json::Value = request
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(|e| e.to_string())?
        .json()
        .await
        .map_err(|e| e.to_string())?;
    props["default_generation_settings"]["n_ctx"]
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| "the server reports no default_generation_settings.n_ctx".to_owned())
}
