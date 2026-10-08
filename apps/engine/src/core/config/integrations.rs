//! Settings for the per-user and public integrations the tools reach: OAuth clients and APIs.

use secrecy::SecretString;
use serde::Deserialize;

/// OAuth clients for per-user grants.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct OAuth {
    /// Google, for Calendar through its MCP server.
    pub google: GoogleOAuth,
    /// Canvas, for a student's own courses, assignments, and grades.
    pub canvas: CanvasOAuth,
    /// Microsoft, for a student's own Outlook calendar and mail.
    pub microsoft: MicrosoftOAuth,
}

/// A Canvas OAuth 2.0 web client for per-user grants. Off by default.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct CanvasOAuth {
    /// Whether the client is built at boot and the login flow is offered.
    pub enabled: bool,
    /// OAuth client id from the Canvas developer key.
    pub client_id: String,
    /// OAuth client secret. Lives in .env.
    pub client_secret: SecretString,
    /// Where Canvas sends the user back with a code.
    pub redirect_url: String,
    /// Scopes requested in one grant. Empty asks for the full access the key allows.
    pub scopes: Vec<String>,
    /// Canvas authorization endpoint.
    pub authorize_url: String,
    /// Canvas token endpoint.
    pub token_url: String,
    /// Budget for one token request.
    pub timeout_secs: u64,
}

impl Default for CanvasOAuth {
    fn default() -> Self {
        Self {
            enabled: false,
            client_id: String::new(),
            client_secret: SecretString::from(""),
            redirect_url: String::new(),
            scopes: Vec::new(),
            authorize_url: "https://canvas.asu.edu/login/oauth2/auth".into(),
            token_url: "https://canvas.asu.edu/login/oauth2/token".into(),
            timeout_secs: 30,
        }
    }
}

/// A Microsoft OAuth 2.0 web client for per-user grants. Off by default.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct MicrosoftOAuth {
    /// Whether the client is built at boot and the login flow is offered.
    pub enabled: bool,
    /// Application (client) id from the Azure app registration.
    pub client_id: String,
    /// OAuth client secret. Lives in .env.
    pub client_secret: SecretString,
    /// Where Microsoft sends the user back with a code.
    pub redirect_url: String,
    /// Scopes requested in one grant. offline_access is needed for refresh.
    pub scopes: Vec<String>,
    /// Microsoft authorization endpoint. The path carries the tenant.
    pub authorize_url: String,
    /// Microsoft token endpoint. The path carries the tenant.
    pub token_url: String,
    /// Budget for one token request.
    pub timeout_secs: u64,
}

impl Default for MicrosoftOAuth {
    fn default() -> Self {
        Self {
            enabled: false,
            client_id: String::new(),
            client_secret: SecretString::from(""),
            redirect_url: String::new(),
            scopes: vec![
                "openid".into(),
                "profile".into(),
                "offline_access".into(),
                "Calendars.Read".into(),
                "Mail.Read".into(),
            ],
            authorize_url: "https://login.microsoftonline.com/common/oauth2/v2.0/authorize".into(),
            token_url: "https://login.microsoftonline.com/common/oauth2/v2.0/token".into(),
            timeout_secs: 30,
        }
    }
}

/// Outlook (Microsoft Graph) read-only tools. Off by default; needs an oauth.microsoft grant.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Outlook {
    /// Whether the Outlook tools are registered at boot.
    pub enabled: bool,
    /// Base URL of Microsoft Graph, no trailing path.
    pub base_url: String,
    /// Budget for one Graph request.
    pub timeout_secs: u64,
    /// Most rows one list tool returns to the model.
    pub max_items: usize,
}

impl Default for Outlook {
    fn default() -> Self {
        Self {
            enabled: false,
            base_url: "https://graph.microsoft.com/v1.0".into(),
            timeout_secs: 30,
            max_items: 15,
        }
    }
}

/// Google Calendar read-only tool. Off by default; needs an oauth.google grant.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Gcal {
    /// Whether the Google Calendar tool is registered at boot.
    pub enabled: bool,
    /// Base URL of the Google Calendar API, no trailing path.
    pub base_url: String,
    /// Budget for one request.
    pub timeout_secs: u64,
    /// Most events one call returns to the model.
    pub max_items: usize,
}

impl Default for Gcal {
    fn default() -> Self {
        Self {
            enabled: false,
            base_url: "https://www.googleapis.com/calendar/v3".into(),
            timeout_secs: 30,
            max_items: 15,
        }
    }
}

/// Academic paper search over Semantic Scholar. No key; a public read.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Papers {
    /// Whether the paper search tool is registered at boot.
    pub enabled: bool,
    /// Base URL of the Semantic Scholar graph API, no trailing path.
    pub base_url: String,
    /// Budget for one request.
    pub timeout_secs: u64,
    /// Most papers returned to the model.
    pub max_items: usize,
}

impl Default for Papers {
    fn default() -> Self {
        Self {
            enabled: true,
            base_url: "https://api.semanticscholar.org/graph/v1".into(),
            timeout_secs: 20,
            max_items: 5,
        }
    }
}

/// Wikipedia lookups over the MediaWiki API. No key; a public read.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Wikipedia {
    /// Whether the Wikipedia tool is registered at boot.
    pub enabled: bool,
    /// The MediaWiki API endpoint.
    pub base_url: String,
    /// Budget for one request.
    pub timeout_secs: u64,
    /// Most characters of the summary returned to the model.
    pub max_chars: usize,
}

impl Default for Wikipedia {
    fn default() -> Self {
        Self {
            enabled: true,
            base_url: "https://en.wikipedia.org/w/api.php".into(),
            timeout_secs: 20,
            max_chars: 1500,
        }
    }
}

/// Valley Metro transit realtime. Off by default; needs a GTFS-realtime JSON feed URL.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Transit {
    /// Whether the transit tool is registered at boot.
    pub enabled: bool,
    /// GTFS-realtime vehicle positions feed that returns JSON. Lives in .env; may carry a key.
    pub feed_url: SecretString,
    /// Static GTFS routes.txt URL, for mapping route ids to names. Empty leaves ids unresolved.
    pub routes_url: String,
    /// Budget for one request.
    pub timeout_secs: u64,
    /// Most rows returned to the model.
    pub max_items: usize,
}

impl Default for Transit {
    fn default() -> Self {
        Self {
            enabled: false,
            feed_url: SecretString::from(""),
            routes_url: String::new(),
            timeout_secs: 20,
            max_items: 20,
        }
    }
}

/// Canvas LMS integration. Off by default. Read-only, in a direct message only.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct Canvas {
    /// Whether the Canvas tools are registered at boot.
    pub enabled: bool,
    /// Base URL of the Canvas instance, no trailing path.
    pub base_url: String,
    /// Canvas API token used until per-user grants exist. Lives in .env.
    pub access_token: SecretString,
    /// Budget for one Canvas API request.
    pub timeout_secs: u64,
    /// Most rows one list tool returns to the model.
    pub max_items: usize,
}

impl Default for Canvas {
    fn default() -> Self {
        Self {
            enabled: false,
            base_url: "https://canvas.asu.edu".into(),
            access_token: SecretString::from(""),
            timeout_secs: 30,
            max_items: 20,
        }
    }
}

/// A Google OAuth 2.0 web client. Off by default.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct GoogleOAuth {
    /// Whether the client is built at boot.
    pub enabled: bool,
    /// OAuth client id from the Google Cloud console.
    pub client_id: String,
    /// OAuth client secret. Lives in .env.
    pub client_secret: SecretString,
    /// Where Google sends the user back with a code.
    pub redirect_url: String,
    /// Scopes requested in one grant.
    pub scopes: Vec<String>,
    /// Google authorization endpoint.
    pub authorize_url: String,
    /// Google token endpoint.
    pub token_url: String,
    /// Budget for one token request.
    pub timeout_secs: u64,
}

impl Default for GoogleOAuth {
    fn default() -> Self {
        Self {
            enabled: false,
            client_id: String::new(),
            client_secret: SecretString::from(""),
            redirect_url: String::new(),
            scopes: vec!["https://www.googleapis.com/auth/calendar.events.readonly".into()],
            authorize_url: "https://accounts.google.com/o/oauth2/v2/auth".into(),
            token_url: "https://oauth2.googleapis.com/token".into(),
            timeout_secs: 30,
        }
    }
}
