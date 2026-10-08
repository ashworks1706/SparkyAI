//! Product events.

/// One product event, exported as a span.
#[derive(Debug, Clone)]
pub struct AnalyticsEvent {
    /// Event name, such as discord_ask.
    pub event: &'static str,
    /// The Discord user id.
    pub distinct_id: String,
    /// Event properties, recorded as span attributes.
    pub properties: serde_json::Map<String, serde_json::Value>,
}

impl AnalyticsEvent {
    /// An event named event for distinct_id, with no properties.
    pub fn new(event: &'static str, distinct_id: &impl ToString) -> Self {
        Self {
            event,
            distinct_id: distinct_id.to_string(),
            properties: serde_json::Map::new(),
        }
    }

    /// Adds one property.
    #[must_use]
    pub fn with(mut self, key: &str, value: impl Into<serde_json::Value>) -> Self {
        self.properties.insert(key.to_owned(), value.into());
        self
    }
}
