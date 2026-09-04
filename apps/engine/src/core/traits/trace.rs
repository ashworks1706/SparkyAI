//! `TraceSink` trait.

use crate::core::types::context::RequestContext;
use crate::core::types::trace::TraceEvent;

/// Where trace events go. Must not block the loop.
pub trait TraceSink: Send + Sync {
    /// Records one event.
    fn emit(&self, ctx: &RequestContext, event: TraceEvent);
}
