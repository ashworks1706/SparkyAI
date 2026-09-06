//! Per-user request limiting for the routes that start an agent run.

use std::sync::Arc;
use std::time::Duration;

/// Fixed-window per-user rate limit, shared by every route that starts a run.
///
/// A window is a whole minute and resets for everyone at once: the point is to stop one caller
/// from monopolising the model, not to meter usage precisely.
#[derive(Clone)]
pub struct RateLimiter {
    per_min: u32,
    state: Arc<std::sync::Mutex<RateWindow>>,
}

struct RateWindow {
    started: std::time::Instant,
    counts: std::collections::HashMap<String, u32>,
}

impl RateLimiter {
    /// Allows `per_min` requests per user per minute. Zero removes the limit.
    pub fn new(per_min: u32) -> Self {
        Self {
            per_min,
            state: Arc::new(std::sync::Mutex::new(RateWindow {
                started: std::time::Instant::now(),
                counts: std::collections::HashMap::new(),
            })),
        }
    }

    /// Counts one request from `user` and says whether it may run.
    pub fn allow(&self, user: &str) -> bool {
        if self.per_min == 0 {
            return true;
        }
        let Ok(mut state) = self.state.lock() else {
            // A poisoned lock means a panic elsewhere. Refusing every request on top of that
            // helps nobody, so the limit opens — but it opens loudly, because a silently
            // disabled rate limit is indistinguishable from one that is working.
            tracing::error!(
                user,
                "rate limiter lock poisoned; requests are unlimited until restart"
            );
            return true;
        };
        if state.started.elapsed() >= Duration::from_mins(1) {
            state.started = std::time::Instant::now();
            state.counts.clear();
        }
        let count = state.counts.entry(user.to_owned()).or_insert(0);
        *count += 1;
        *count <= self.per_min
    }
}

impl std::fmt::Debug for RateLimiter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RateLimiter")
            .field("per_min", &self.per_min)
            .finish_non_exhaustive()
    }
}
