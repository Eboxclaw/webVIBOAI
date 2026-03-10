/// oauth.rs — ViBo Generic OAuth 2.0 + PKCE
///
/// Handles the full OAuth lifecycle for any provider.
/// Tokens stored encrypted in crypto keystore.
///
/// Usage (any provider):
///   oauth_register_provider(name, config)   → register once
///   oauth_auth_start(provider_name)         → returns consent URL
///   oauth_auth_callback(provider, code)     → exchanges code, stores tokens
///   oauth_get_token(provider_name)          → returns valid token (auto-refresh)
///   oauth_revoke(provider_name)             → removes all tokens
///
/// Keystore keys per provider:
///   "{provider}_access_token"
///   "{provider}_refresh_token"
///   "{provider}_token_expiry"
///   "{provider}_client_id"
///   "{provider}_client_secret"
///
/// Cargo.toml:
///   reqwest, serde, serde_json, url, chrono, base64, sha2, rand

use base64::engine::general_purpose::{URL_SAFE_NO_PAD, STANDARD as B64};
use base64::Engine;
use chrono::Utc;
use rand::RngCore;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Mutex;
use tauri::State;
use url::Url;

// ─────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────

/// Provider config — registered once at setup or by user
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OAuthProviderConfig {
    pub name: String,               // e.g. "google", "notion", "github"
    pub auth_url: String,           // consent page URL
    pub token_url: String,          // token exchange endpoint
    pub redirect_uri: String,       // deep link e.g. "vibo://oauth/google/callback"
    pub scopes: Vec<String>,
    pub pkce: bool,                 // use PKCE (recommended, required for mobile)
    pub extra_params: HashMap<String, String>, // e.g. {"access_type": "offline"}
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OAuthStatus {
    pub provider: String,
    pub authenticated: bool,
    pub token_expires_at: Option<String>,
    pub scopes: Vec<String>,
}

/// In-memory PKCE verifier — stored between auth_start and auth_callback
struct PkceState {
    code_verifier: String,
    provider: String,
}

pub struct OAuthState {
    providers: Mutex<HashMap<String, OAuthProviderConfig>>,
    /// Pending PKCE verifiers keyed by state param
    pending: Mutex<HashMap<String, PkceState>>,
}

impl OAuthState {
    pub fn new() -> Self {
        OAuthState {
            providers: Mutex::new(HashMap::new()),
            pending: Mutex::new(HashMap::new()),
        }
    }
}

// ─────────────────────────────────────────
// PKCE HELPERS
// ─────────────────────────────────────────

fn generate_code_verifier() -> String {
    let mut bytes = [0u8; 64];
    rand::thread_rng().fill_bytes(&mut bytes);
    URL_SAFE_NO_PAD.encode(bytes)
}

fn generate_code_challenge(verifier: &str) -> String {
    let hash = Sha256::digest(verifier.as_bytes());
    URL_SAFE_NO_PAD.encode(hash)
}

fn generate_state_param() -> String {
    let mut bytes = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut bytes);
    URL_SAFE_NO_PAD.encode(bytes)
}

// ─────────────────────────────────────────
// KEYSTORE HELPERS
// ─────────────────────────────────────────

fn token_key(provider: &str, kind: &str) -> String {
    format!("{}_{}",  provider, kind)
}

fn store_token(crypto: &crate::crypto::CryptoState, provider: &str, kind: &str, value: &str) -> Result<(), String> {
    crate::crypto::crypto_keystore_set(
        unsafe { std::mem::transmute(crypto) },
        token_key(provider, kind),
        value.to_string(),
        Some("oauth_token".to_string()),
    )
}

fn load_token(crypto: &crate::crypto::CryptoState, provider: &str, kind: &str) -> Result<String, String> {
    crate::crypto::crypto_keystore_get(
        unsafe { std::mem::transmute(crypto) },
        token_key(provider, kind),
    ).map_err(|_| format!("{} not authenticated — call oauth_auth_start first", provider))
}

fn delete_token(crypto: &crate::crypto::CryptoState, provider: &str, kind: &str) {
    let _ = crate::crypto::crypto_keystore_delete(
        unsafe { std::mem::transmute(crypto) },
        token_key(provider, kind),
    );
}

fn is_token_expired(crypto: &crate::crypto::CryptoState, provider: &str) -> bool {
    let Ok(expiry_str) = load_token(crypto, provider, "token_expiry") else { return true };
    let Ok(expiry) = chrono::DateTime::parse_from_rfc3339(&expiry_str) else { return true };
    Utc::now() > expiry.with_timezone(&Utc) - chrono::Duration::seconds(60)
}

// ─────────────────────────────────────────
// PROVIDER REGISTRATION
// ─────────────────────────────────────────

/// Register a provider config — call once at app setup or from settings
#[tauri::command]
pub fn oauth_register_provider(
    state: State<OAuthState>,
    config: OAuthProviderConfig,
) -> Result<(), String> {
    state.providers.lock().unwrap()
        .insert(config.name.clone(), config);
    Ok(())
}

/// Store client credentials for a provider (from user settings)
#[tauri::command]
pub fn oauth_set_credentials(
    state: State<OAuthState>,
    crypto: State<crate::crypto::CryptoState>,
    provider: String,
    client_id: String,
    client_secret: String,
) -> Result<(), String> {
    // Verify provider is registered
    if !state.providers.lock().unwrap().contains_key(&provider) {
        return Err(format!("Provider '{}' not registered — call oauth_register_provider first", provider));
    }
    store_token(&crypto, &provider, "client_id", &client_id)?;
    store_token(&crypto, &provider, "client_secret", &client_secret)?;
    Ok(())
}

/// List all registered providers
#[tauri::command]
pub fn oauth_list_providers(state: State<OAuthState>) -> Vec<String> {
    state.providers.lock().unwrap().keys().cloned().collect()
}

// ─────────────────────────────────────────
// AUTH FLOW
// ─────────────────────────────────────────

/// Start OAuth flow — returns consent URL to open in browser/webview
/// Generates PKCE verifier if provider requires it
#[tauri::command]
pub fn oauth_auth_start(
    state: State<OAuthState>,
    crypto: State<crate::crypto::CryptoState>,
    provider: String,
) -> Result<String, String> {
    let providers = state.providers.lock().unwrap();
    let config = providers.get(&provider)
        .ok_or_else(|| format!("Provider '{}' not registered", provider))?
        .clone();
    drop(providers);

    let client_id = load_token(&crypto, &provider, "client_id")?;
    let scope = config.scopes.join(" ");
    let state_param = generate_state_param();

    let mut url = Url::parse(&config.auth_url)
        .map_err(|e| format!("Invalid auth_url: {}", e))?;

    {
        let mut params = url.query_pairs_mut();
        params.append_pair("client_id", &client_id);
        params.append_pair("redirect_uri", &config.redirect_uri);
        params.append_pair("response_type", "code");
        params.append_pair("scope", &scope);
        params.append_pair("state", &state_param);

        // Extra provider-specific params (e.g. access_type=offline for Google)
        for (k, v) in &config.extra_params {
            params.append_pair(k, v);
        }

        if config.pkce {
            let verifier = generate_code_verifier();
            let challenge = generate_code_challenge(&verifier);
            params.append_pair("code_challenge", &challenge);
            params.append_pair("code_challenge_method", "S256");

            // Store verifier keyed by state param
            drop(params);
            state.pending.lock().unwrap().insert(state_param.clone(), PkceState {
                code_verifier: verifier,
                provider: provider.clone(),
            });
        }
    }

    Ok(url.to_string())
}

/// Complete OAuth flow — exchange code for tokens
/// state_param must match what was returned by oauth_auth_start
#[tauri::command]
pub async fn oauth_auth_callback(
    oauth_state: State<'_, OAuthState>,
    crypto: State<'_, crate::crypto::CryptoState>,
    providers_state: State<'_, crate::providers::ProvidersState>,
    provider: String,
    code: String,
    state_param: Option<String>,    // PKCE state verification
) -> Result<OAuthStatus, String> {
    let config = oauth_state.providers.lock().unwrap()
        .get(&provider)
        .ok_or_else(|| format!("Provider '{}' not registered", provider))?
        .clone();

    let client_id     = load_token(&crypto, &provider, "client_id")?;
    let client_secret = load_token(&crypto, &provider, "client_secret")?;

    // Retrieve PKCE verifier if applicable
    let code_verifier = if config.pkce {
        let key = state_param.ok_or("state_param required for PKCE")?;
        let pkce = oauth_state.pending.lock().unwrap()
            .remove(&key)
            .ok_or("PKCE state not found — auth flow may have expired")?;
        if pkce.provider != provider {
            return Err("PKCE state provider mismatch".to_string());
        }
        Some(pkce.code_verifier)
    } else {
        None
    };

    // Build token exchange request
    let tor_on = *providers_state.tor_enabled.lock().unwrap();
    let mut builder = Client::builder()
        .timeout(std::time::Duration::from_secs(30));
    if tor_on {
        let proxy = reqwest::Proxy::all(&providers_state.tor_proxy)
            .map_err(|e| e.to_string())?;
        builder = builder.proxy(proxy);
    }
    let client = builder.build().map_err(|e| e.to_string())?;

    let mut form = vec![
        ("grant_type",    "authorization_code".to_string()),
        ("code",          code),
        ("redirect_uri",  config.redirect_uri.clone()),
        ("client_id",     client_id),
        ("client_secret", client_secret),
    ];
    if let Some(verifier) = code_verifier {
        form.push(("code_verifier", verifier));
    }

    let response = client
        .post(&config.token_url)
        .form(&form)
        .send()
        .await
        .map_err(|e| format!("Token exchange failed: {}", e))?;

    let json: Value = response.json().await.map_err(|e| e.to_string())?;

    if let Some(error) = json["error"].as_str() {
        return Err(format!("OAuth error: {} — {}",
            error, json["error_description"].as_str().unwrap_or("")));
    }

    let access_token  = json["access_token"].as_str().ok_or("No access_token")?;
    let expires_in    = json["expires_in"].as_i64().unwrap_or(3600);
    let expiry        = Utc::now() + chrono::Duration::seconds(expires_in);

    store_token(&crypto, &provider, "access_token", access_token)?;
    store_token(&crypto, &provider, "token_expiry", &expiry.to_rfc3339())?;

    // refresh_token only comes on first auth (not on refresh)
    if let Some(rt) = json["refresh_token"].as_str() {
        store_token(&crypto, &provider, "refresh_token", rt)?;
    }

    Ok(OAuthStatus {
        provider,
        authenticated: true,
        token_expires_at: Some(expiry.to_rfc3339()),
        scopes: config.scopes,
    })
}

// ─────────────────────────────────────────
// TOKEN ACCESS (used by other modules)
// ─────────────────────────────────────────

/// Get a valid access token — auto-refreshes if expired
/// Called internally by google.rs, notion.rs, etc.
#[tauri::command]
pub async fn oauth_get_token(
    oauth_state: State<'_, OAuthState>,
    crypto: State<'_, crate::crypto::CryptoState>,
    providers_state: State<'_, crate::providers::ProvidersState>,
    provider: String,
) -> Result<String, String> {
    if !is_token_expired(&crypto, &provider) {
        return load_token(&crypto, &provider, "access_token");
    }
    refresh_token_internal(&oauth_state, &crypto, &providers_state, &provider).await
}

/// Internal refresh — called by oauth_get_token and other modules
pub async fn refresh_token_internal(
    oauth_state: &OAuthState,
    crypto: &crate::crypto::CryptoState,
    providers_state: &crate::providers::ProvidersState,
    provider: &str,
) -> Result<String, String> {
    let config = oauth_state.providers.lock().unwrap()
        .get(provider)
        .ok_or_else(|| format!("Provider '{}' not registered", provider))?
        .clone();

    let refresh_token = load_token(crypto, provider, "refresh_token")?;
    let client_id     = load_token(crypto, provider, "client_id")?;
    let client_secret = load_token(crypto, provider, "client_secret")?;

    let tor_on = *providers_state.tor_enabled.lock().unwrap();
    let mut builder = Client::builder()
        .timeout(std::time::Duration::from_secs(30));
    if tor_on {
        let proxy = reqwest::Proxy::all(&providers_state.tor_proxy)
            .map_err(|e| e.to_string())?;
        builder = builder.proxy(proxy);
    }
    let client = builder.build().map_err(|e| e.to_string())?;

    let response = client
        .post(&config.token_url)
        .form(&[
            ("grant_type",    "refresh_token"),
            ("refresh_token", &refresh_token),
            ("client_id",     &client_id),
            ("client_secret", &client_secret),
        ])
        .send()
        .await
        .map_err(|e| format!("Token refresh failed: {}", e))?;

    let json: Value = response.json().await.map_err(|e| e.to_string())?;

    if let Some(error) = json["error"].as_str() {
        return Err(format!("Refresh error: {}", error));
    }

    let access_token = json["access_token"].as_str()
        .ok_or("No access_token in refresh response")?;
    let expires_in   = json["expires_in"].as_i64().unwrap_or(3600);
    let expiry       = Utc::now() + chrono::Duration::seconds(expires_in);

    store_token(crypto, provider, "access_token", access_token)?;
    store_token(crypto, provider, "token_expiry", &expiry.to_rfc3339())?;

    Ok(access_token.to_string())
}

// ─────────────────────────────────────────
// STATUS + REVOKE
// ─────────────────────────────────────────

/// Check auth status for a provider
#[tauri::command]
pub fn oauth_status(
    oauth_state: State<OAuthState>,
    crypto: State<crate::crypto::CryptoState>,
    provider: String,
) -> OAuthStatus {
    let authenticated = load_token(&crypto, &provider, "refresh_token").is_ok();
    let token_expires_at = load_token(&crypto, &provider, "token_expiry").ok();
    let scopes = oauth_state.providers.lock().unwrap()
        .get(&provider)
        .map(|c| c.scopes.clone())
        .unwrap_or_default();

    OAuthStatus { provider, authenticated, token_expires_at, scopes }
}

/// Revoke all tokens for a provider
#[tauri::command]
pub fn oauth_revoke(
    crypto: State<crate::crypto::CryptoState>,
    provider: String,
) -> Result<(), String> {
    delete_token(&crypto, &provider, "access_token");
    delete_token(&crypto, &provider, "refresh_token");
    delete_token(&crypto, &provider, "token_expiry");
    Ok(())
}

// ─────────────────────────────────────────
// BUILT-IN PROVIDER CONFIGS
// Called from main.rs setup to pre-register known providers
// ─────────────────────────────────────────

pub fn google_provider_config() -> OAuthProviderConfig {
    OAuthProviderConfig {
        name: "google".to_string(),
        auth_url: "https://accounts.google.com/o/oauth2/v2/auth".to_string(),
        token_url: "https://oauth2.googleapis.com/token".to_string(),
        redirect_uri: "vibo://oauth/google/callback".to_string(),
        scopes: vec![
            "https://www.googleapis.com/auth/calendar".to_string(),
            "https://www.googleapis.com/auth/gmail.readonly".to_string(),
        ],
        pkce: true,
        extra_params: [
            ("access_type".to_string(), "offline".to_string()),
            ("prompt".to_string(), "consent".to_string()),
        ].into(),
    }
}

// Future providers — uncomment when needed:
//
// pub fn notion_provider_config() -> OAuthProviderConfig {
//     OAuthProviderConfig {
//         name: "notion".to_string(),
//         auth_url: "https://api.notion.com/v1/oauth/authorize".to_string(),
//         token_url: "https://api.notion.com/v1/oauth/token".to_string(),
//         redirect_uri: "vibo://oauth/notion/callback".to_string(),
//         scopes: vec!["read_content".to_string(), "update_content".to_string()],
//         pkce: false,
//         extra_params: HashMap::new(),
//     }
// }
//
// pub fn github_provider_config() -> OAuthProviderConfig {
//     OAuthProviderConfig {
//         name: "github".to_string(),
//         auth_url: "https://github.com/login/oauth/authorize".to_string(),
//         token_url: "https://github.com/login/oauth/access_token".to_string(),
//         redirect_uri: "vibo://oauth/github/callback".to_string(),
//         scopes: vec!["repo".to_string(), "read:user".to_string()],
//         pkce: true,
//         extra_params: HashMap::new(),
//     }
// }

// ─────────────────────────────────────────
// REGISTER in main.rs:
//
// .invoke_handler(tauri::generate_handler![
//     oauth::oauth_register_provider,
//     oauth::oauth_set_credentials,
//     oauth::oauth_list_providers,
//     oauth::oauth_auth_start,
//     oauth::oauth_auth_callback,
//     oauth::oauth_get_token,
//     oauth::oauth_status,
//     oauth::oauth_revoke,
// ])
//
// In setup():
//   let oauth_state = oauth::OAuthState::new();
//   oauth_state.providers.lock().unwrap()
//       .insert("google".to_string(), oauth::google_provider_config());
//   app.manage(oauth_state);
// ─────────────────────────────────────────
