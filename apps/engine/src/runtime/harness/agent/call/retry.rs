//! How long the loop waits between model retries.

use std::time::Duration;
use uuid::Uuid;

/// Wait before retry: doubles from base_ms, capped at cap_ms, jittered, never past the deadline.
pub fn backoff(
    attempt: u32,
    request_id: Uuid,
    remaining: Duration,
    base_ms: u64,
    cap_ms: u64,
) -> Duration {
    let doubled = base_ms.saturating_mul(1u64 << attempt.min(6)).min(cap_ms);
    let spread = doubled / 4;
    #[allow(clippy::cast_possible_truncation)]
    let offset = (request_id.as_u128() as u64) % spread.max(1);
    Duration::from_millis(doubled - spread / 2 + offset).min(remaining)
}
