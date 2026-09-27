//! Google OAuth 2.0: the consent URL, the code exchange, and the refresh of a per-user grant.

use std::time::Duration;

use chrono::{TimeDelta, Utc};
use secrecy::{ExposeSecret, SecretString};
use serde::Deserialize;
use url::Url;

use crate::core::config::GoogleOAuth;
use crate::core::types::tools::oauth::{OAuthError, OAuthTokens};

/// Longest OAuth error code repeated from a response.
const MAX_CODE_CHARS: usize = 64;

/// A token endpoint response. Fields Google may leave out are optional.
#[derive(Deserialize)]
struct TokenResponse {
    access_token: Option<String>,
    refresh_token: Option<String>,
    scope: Option<String>,
    expires_in: Option<i64>,
}

/// A token endpoint error body.
#[derive(Deserialize)]
struct ErrorResponse {
    error: Option<String>,
}

/// A Google OAuth web client for per-user grants with offline access.
pub struct GoogleOAuthClient {
    http: reqwest::Client,
    client_id: String,
    client_secret: SecretString,
    redirect_url: String,
    scopes: Vec<String>,
    authorize_url: Url,
    token_url: Url,
}

impl GoogleOAuthClient {
    /// Builds the client from its settings.
    ///
    /// # Errors
    /// `NotConfigured` when an endpoint is not a URL or the HTTP client cannot be built.
    pub fn new(cfg: &GoogleOAuth) -> Result<Self, OAuthError> {
        let parse = |s: &str| Url::parse(s).map_err(|e| OAuthError::NotConfigured(e.to_string()));
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(cfg.timeout_secs))
            .build()
            .map_err(|e| OAuthError::NotConfigured(e.to_string()))?;
        Ok(Self {
            http,
            client_id: cfg.client_id.clone(),
            client_secret: cfg.client_secret.clone(),
            redirect_url: cfg.redirect_url.clone(),
            scopes: cfg.scopes.clone(),
            authorize_url: parse(&cfg.authorize_url)?,
            token_url: parse(&cfg.token_url)?,
        })
    }

    /// The consent URL the user opens. state binds the callback to this request.
    #[must_use]
    pub fn authorize_url(&self, state: &str) -> String {
        let mut url = self.authorize_url.clone();
        url.query_pairs_mut()
            .append_pair("client_id", &self.client_id)
            .append_pair("redirect_uri", &self.redirect_url)
            .append_pair("response_type", "code")
            .append_pair("scope", &self.scopes.join(" "))
            .append_pair("state", state)
            .append_pair("access_type", "offline")
            .append_pair("prompt", "consent")
            .append_pair("include_granted_scopes", "true");
        url.into()
    }

    /// Tokens for an authorization code.
    ///
    /// # Errors
    /// Any `OAuthError`; `NoRefreshToken` when the grant is not offline.
    pub async fn exchange(&self, code: &str) -> Result<OAuthTokens, OAuthError> {
        let tokens = self
            .token(&[
                ("grant_type", "authorization_code"),
                ("code", code),
                ("redirect_uri", &self.redirect_url),
            ])
            .await
            .and_then(|r| tokens(r, None))?;
        if tokens.refresh_token.is_none() {
            return Err(OAuthError::NoRefreshToken);
        }
        Ok(tokens)
    }

    /// New tokens for a grant, keeping what the response leaves out.
    ///
    /// # Errors
    /// Any `OAuthError`; `NoRefreshToken` when the grant has none.
    pub async fn refresh(&self, previous: &OAuthTokens) -> Result<OAuthTokens, OAuthError> {
        let refresh = previous
            .refresh_token
            .as_ref()
            .ok_or(OAuthError::NoRefreshToken)?;
        let response = self
            .token(&[
                ("grant_type", "refresh_token"),
                ("refresh_token", refresh.expose_secret()),
            ])
            .await?;
        tokens(response, Some(previous))
    }

    async fn token(&self, form: &[(&str, &str)]) -> Result<TokenResponse, OAuthError> {
        let mut body = url::form_urlencoded::Serializer::new(String::new());
        body.extend_pairs(form);
        body.append_pair("client_id", &self.client_id);
        body.append_pair("client_secret", self.client_secret.expose_secret());
        let response = self
            .http
            .post(self.token_url.clone())
            .header(
                reqwest::header::CONTENT_TYPE,
                "application/x-www-form-urlencoded",
            )
            .body(body.finish())
            .send()
            .await
            .map_err(|e| OAuthError::Unreachable(kind_of(&e)))?;
        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|e| OAuthError::Unreachable(kind_of(&e)))?;
        if !status.is_success() {
            return Err(OAuthError::Refused {
                status: status.as_u16(),
                code: error_code(&text),
            });
        }
        serde_json::from_str(&text).map_err(|_| OAuthError::Malformed("not a token object".into()))
    }
}

/// A transport failure named by its kind, never by its URL or body.
fn kind_of(e: &reqwest::Error) -> String {
    if e.is_timeout() {
        "timeout".into()
    } else if e.is_connect() {
        "connect".into()
    } else {
        "request".into()
    }
}

/// The OAuth error code of a body when it is a short identifier, else empty.
fn error_code(body: &str) -> String {
    serde_json::from_str::<ErrorResponse>(body)
        .ok()
        .and_then(|r| r.error)
        .filter(|c| {
            c.len() <= MAX_CODE_CHARS && c.bytes().all(|b| b.is_ascii_lowercase() || b == b'_')
        })
        .unwrap_or_default()
}

/// Tokens from a response, keeping the refresh token and scopes of the previous grant.
fn tokens(r: TokenResponse, previous: Option<&OAuthTokens>) -> Result<OAuthTokens, OAuthError> {
    let access = r
        .access_token
        .filter(|t| !t.is_empty())
        .ok_or_else(|| OAuthError::Malformed("no access token".into()))?;
    let refresh_token = r
        .refresh_token
        .filter(|t| !t.is_empty())
        .map(SecretString::from)
        .or_else(|| previous.and_then(|p| p.refresh_token.clone()));
    let scopes = match r.scope.filter(|s| !s.is_empty()) {
        Some(s) => s.split_whitespace().map(str::to_owned).collect(),
        None => previous.map(|p| p.scopes.clone()).unwrap_or_default(),
    };
    let expires_at = r
        .expires_in
        .and_then(TimeDelta::try_seconds)
        .map(|d| Utc::now() + d);
    Ok(OAuthTokens {
        access_token: SecretString::from(access),
        refresh_token,
        scopes,
        expires_at,
    })
}
