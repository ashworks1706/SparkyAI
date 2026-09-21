//! FileSource trait.

use async_trait::async_trait;

use crate::core::types::agent::context::RequestContext;
use crate::core::types::conversation::file::{FileAttachment, FileError};

/// Downloads a file the caller attached.
#[async_trait]
pub trait FileSource: Send + Sync {
    /// The bytes of file, or why they could not be read.
    async fn fetch(
        &self,
        ctx: &RequestContext,
        file: &FileAttachment,
    ) -> Result<Vec<u8>, FileError>;
}
