//! What the loop decides about knowledge before it reaches a store: the gate on retrieval, the
//! cache in front of live queries, and the cap on how many of them run at once.

pub mod admit;
pub mod cache;
pub mod route;
