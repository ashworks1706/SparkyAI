//! Probe doubles: a dependency that is up or down.

use async_trait::async_trait;

use crate::core::traits::health::Probe;

/// A probe that always answers the same.
pub struct Fixed {
    /// Its name in the report.
    pub name: &'static str,
    /// Whether it answers.
    pub up: bool,
}

#[async_trait]
impl Probe for Fixed {
    fn name(&self) -> &'static str {
        self.name
    }

    async fn ready(&self) -> bool {
        self.up
    }
}
