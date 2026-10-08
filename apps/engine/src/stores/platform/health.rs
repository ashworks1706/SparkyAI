//! Readiness of the platform.

use async_trait::async_trait;

use super::client::PlatformClient;
use crate::core::traits::health::Probe;

/// Answers readiness from the platform health route.
pub struct PlatformProbe {
    client: PlatformClient,
}

impl PlatformProbe {
    /// Wraps a client.
    pub fn new(client: PlatformClient) -> Self {
        Self { client }
    }
}

#[async_trait]
impl Probe for PlatformProbe {
    fn name(&self) -> &'static str {
        "platform"
    }

    async fn ready(&self) -> bool {
        self.client.ready().await
    }
}
