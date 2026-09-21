//! Downloads the files a caller attached, from the allowed hosts only and within a byte cap.

use async_trait::async_trait;
use futures::StreamExt;

use crate::core::traits::tools::files::FileSource;
use crate::core::types::agent::context::RequestContext;
use crate::core::types::conversation::file::{FileAttachment, FileError};

/// Fetches attachments over HTTPS from a fixed set of hosts.
pub struct HttpFiles {
    client: reqwest::Client,
    hosts: Vec<String>,
    max_bytes: u64,
}

impl HttpFiles {
    /// A fetcher reading from hosts, refusing anything over max_bytes.
    pub fn new(hosts: Vec<String>, max_bytes: u64) -> Result<Self, FileError> {
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|e| FileError::Fetch(e.to_string()))?;
        Ok(Self {
            client,
            hosts: hosts.into_iter().map(|h| h.to_lowercase()).collect(),
            max_bytes,
        })
    }

    fn allowed(&self, link: &str) -> Result<(), FileError> {
        let parsed = url::Url::parse(link).map_err(|e| FileError::Fetch(e.to_string()))?;
        let host = parsed.host_str().unwrap_or_default().to_lowercase();
        if parsed.scheme() != "https" || !self.hosts.contains(&host) {
            return Err(FileError::Host(host));
        }
        Ok(())
    }
}

#[async_trait]
impl FileSource for HttpFiles {
    async fn fetch(
        &self,
        ctx: &RequestContext,
        file: &FileAttachment,
    ) -> Result<Vec<u8>, FileError> {
        self.allowed(&file.url)?;
        let response = self
            .client
            .get(&file.url)
            .timeout(ctx.remaining())
            .send()
            .await
            .and_then(reqwest::Response::error_for_status)
            .map_err(|e| FileError::Fetch(e.to_string()))?;
        if response
            .content_length()
            .is_some_and(|n| n > self.max_bytes)
        {
            return Err(FileError::TooLarge(self.max_bytes));
        }
        let mut body = Vec::new();
        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| FileError::Fetch(e.to_string()))?;
            if (body.len() + chunk.len()) as u64 > self.max_bytes {
                return Err(FileError::TooLarge(self.max_bytes));
            }
            body.extend_from_slice(&chunk);
        }
        Ok(body)
    }
}
