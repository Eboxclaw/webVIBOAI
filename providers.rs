/// providers.rs — ViBo Inference Providers
///
/// Providers:
///   Local:  LFM via Leap SDK (Android/iOS), Ollama (desktop)
///   Cloud:  Anthropic, OpenRouter, Kimi, Minimax + any custom OpenAI-compatible URL
///
/// Streaming: Rust emits Tauri events → frontend listens
///   Events:
///     "llm-delta"    { request_id, delta }       → token chunk
///     "llm-done"     { request_id }              → stream complete
///     "llm-error"    { request_id, error }       → stream failed
///
/// Tor: global on/off toggle, only wraps cloud providers
///      Ollama and LFM always bypass Tor (local only)
///
/// API keys: never stored here — fetched from crypto::crypto_keystore_get
///
/// Cargo.toml dependencies:
///   reqwest = { version = "0.12", features = ["stream", "socks"] }
///   serde = { version = "1", features = ["derive"] }
///   serde_json = "1"
///   tokio = { version = "1", features = ["full"] }
///   futures-util = "0.3"
///   uuid = { version = "1", features = ["v4"] }

use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, State};
use uuid::Uuid;

// ─────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    Leap,           // LFM on-device via Leap SDK (mobile)
    Ollama,         // local desktop
    Anthropic,
    OpenRouter,
    Kimi,
    Minimax,
    Custom,         // any OpenAI-compatible URL
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Message {
    pub role: String,       // "user" | "assistant" | "system"
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CompletionRequest {
    pub provider: ProviderKind,
    pub model: String,
    pub messages: Vec<Message>,
    pub system: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    /// For Custom provider: user-supplied base URL e.g. "https://my-llm.com/v1"
    pub api_url: Option<String>,
    /// Key name in crypto keystore e.g. "anthropic_api_key"
    pub api_key_name: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StreamDelta {
    pub request_id: String,
    pub delta: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StreamDone {
    pub request_id: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StreamError {
    pub request_id: String,
    pub error: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub kind: ProviderKind,
    pub display_name: String,
    pub api_url: String,
    pub default_model: String,
    pub supports_system_prompt: bool,
    pub local: bool,
}

pub struct ProvidersState {
    /// Global Tor toggle — only affects cloud providers
    pub tor_enabled: Mutex<bool>,
    /// Tor SOCKS5 proxy address (local Tor daemon)
    pub tor_proxy: String,
}

impl Default for ProvidersState {
    fn default() -> Self {
        ProvidersState {
            tor_enabled: Mutex::new(false),
            tor_proxy: "socks5h://127.0.0.1:9050".to_string(),
        }
    }
}

// ─────────────────────────────────────────
// HTTP CLIENT BUILDER
// ─────────────────────────────────────────

fn build_client(use_tor: bool, tor_proxy: &str) -> Result<Client, String> {
    let mut builder = Client::builder()
        .timeout(std::time::Duration::from_secs(120));

    if use_tor {
        let proxy = reqwest::Proxy::all(tor_proxy)
            .map_err(|e| format!("Tor proxy error: {}", e))?;
        builder = builder.proxy(proxy);
    }

    builder.build().map_err(|e| e.to_string())
}

fn should_use_tor(state: &ProvidersState, provider: &ProviderKind) -> bool {
    let tor_on = *state.tor_enabled.lock().unwrap();
    if !tor_on { return false; }
    // Never route local providers through Tor
    matches!(provider, ProviderKind::Anthropic | ProviderKind::OpenRouter | ProviderKind::Kimi | ProviderKind::Minimax | ProviderKind::Custom)
}

// ─────────────────────────────────────────
// PROVIDER CONFIGS
// ─────────────────────────────────────────

#[tauri::command]
pub fn providers_list() -> Vec<ProviderConfig> {
    vec![
        ProviderConfig {
            kind: ProviderKind::Leap,
            display_name: "LFM (On-device)".to_string(),
            api_url: "local://leap".to_string(),
            default_model: "lfm2-1.2b".to_string(),
            supports_system_prompt: true,
            local: true,
        },
        ProviderConfig {
            kind: ProviderKind::Ollama,
            display_name: "Ollama (Local)".to_string(),
            api_url: "http://localhost:11434/v1".to_string(),
            default_model: "llama3".to_string(),
            supports_system_prompt: true,
            local: true,
        },
        ProviderConfig {
            kind: ProviderKind::Anthropic,
            display_name: "Anthropic".to_string(),
            api_url: "https://api.anthropic.com/v1/messages".to_string(),
            default_model: "claude-sonnet-4-20250514".to_string(),
            supports_system_prompt: true,
            local: false,
        },
        ProviderConfig {
            kind: ProviderKind::OpenRouter,
            display_name: "OpenRouter".to_string(),
            api_url: "https://openrouter.ai/api/v1/chat/completions".to_string(),
            default_model: "meta-llama/llama-3.1-8b-instruct".to_string(),
            supports_system_prompt: true,
            local: false,
        },
        ProviderConfig {
            kind: ProviderKind::Kimi,
            display_name: "Kimi".to_string(),
            api_url: "https://api.moonshot.cn/v1/chat/completions".to_string(),
            default_model: "moonshot-v1-8k".to_string(),
            supports_system_prompt: true,
            local: false,
        },
        ProviderConfig {
            kind: ProviderKind::Minimax,
            display_name: "Minimax".to_string(),
            api_url: "https://api.minimax.chat/v1/text/chatcompletion_v2".to_string(),
            default_model: "abab6.5s-chat".to_string(),
            supports_system_prompt: true,
            local: false,
        },
        ProviderConfig {
            kind: ProviderKind::Custom,
            display_name: "Custom (OpenAI-compatible)".to_string(),
            api_url: "".to_string(),
            default_model: "".to_string(),
            supports_system_prompt: true,
            local: false,
        },
    ]
}

// ─────────────────────────────────────────
// TOR COMMANDS
// ─────────────────────────────────────────

#[tauri::command]
pub fn providers_tor_set(state: State<ProvidersState>, enabled: bool) -> Result<(), String> {
    let mut tor = state.tor_enabled.lock().unwrap();
    *tor = enabled;
    Ok(())
}

#[tauri::command]
pub fn providers_tor_status(state: State<ProvidersState>) -> bool {
    *state.tor_enabled.lock().unwrap()
}

// ─────────────────────────────────────────
// MAIN STREAM COMMAND
// ─────────────────────────────────────────

/// Main entry point — routes to correct provider, streams back via Tauri events
/// Returns request_id so frontend can match events
#[tauri::command]
pub async fn providers_stream(
    app: AppHandle,
    state: State<'_, ProvidersState>,
    crypto_state: State<'_, crate::crypto::CryptoState>,
    request: CompletionRequest,
) -> Result<String, String> {
    let request_id = Uuid::new_v4().to_string();
    let rid = request_id.clone();
    let use_tor = should_use_tor(&state, &request.provider);
    let tor_proxy = state.tor_proxy.clone();

    // Fetch API key from keystore if needed
    let api_key = if let Some(key_name) = &request.api_key_name {
        Some(crate::crypto::crypto_keystore_get(crypto_state, key_name.clone())?)
    } else {
        None
    };

    // Spawn async stream task
    tokio::spawn(async move {
        let result = match request.provider {
            ProviderKind::Leap => stream_leap(&app, &rid, &request).await,
            ProviderKind::Ollama => {
                let client = build_client(false, &tor_proxy)?; // never Tor for Ollama
                stream_openai_compatible(&app, &rid, &request, &client,
                    "http://localhost:11434/v1/chat/completions", None).await
            }
            ProviderKind::Anthropic => {
                let client = build_client(use_tor, &tor_proxy)?;
                stream_anthropic(&app, &rid, &request, &client,
                    api_key.as_deref()).await
            }
            ProviderKind::OpenRouter => {
                let client = build_client(use_tor, &tor_proxy)?;
                stream_openai_compatible(&app, &rid, &request, &client,
                    "https://openrouter.ai/api/v1/chat/completions",
                    api_key.as_deref()).await
            }
            ProviderKind::Kimi => {
                let client = build_client(use_tor, &tor_proxy)?;
                stream_openai_compatible(&app, &rid, &request, &client,
                    "https://api.moonshot.cn/v1/chat/completions",
                    api_key.as_deref()).await
            }
            ProviderKind::Minimax => {
                let client = build_client(use_tor, &tor_proxy)?;
                stream_openai_compatible(&app, &rid, &request, &client,
                    "https://api.minimax.chat/v1/text/chatcompletion_v2",
                    api_key.as_deref()).await
            }
            ProviderKind::Custom => {
                let client = build_client(use_tor, &tor_proxy)?;
                let url = request.api_url.clone()
                    .ok_or("Custom provider requires api_url".to_string())?;
                stream_openai_compatible(&app, &rid, &request, &client,
                    &url, api_key.as_deref()).await
            }
        };

        if let Err(e) = result {
            let _ = app.emit("llm-error", StreamError {
                request_id: rid.clone(),
                error: e,
            });
        }

        Ok::<(), String>(())
    });

    Ok(request_id)
}

// ─────────────────────────────────────────
// PROVIDER IMPLEMENTATIONS
// ─────────────────────────────────────────

/// LFM via Leap SDK — delegates to Kotlin sidecar via IPC
/// The Kotlin side emits tokens back through Tauri events directly
async fn stream_leap(
    app: &AppHandle,
    request_id: &str,
    request: &CompletionRequest,
) -> Result<(), String> {
    // Kotlin LeapPlugin handles the actual LFM inference
    // We just forward the request payload to the Kotlin sidecar
    // Kotlin emits "llm-delta" and "llm-done" events directly
    app.emit("leap-infer-request", json!({
        "request_id": request_id,
        "model": request.model,
        "messages": request.messages,
        "system": request.system,
        "max_tokens": request.max_tokens.unwrap_or(1024),
        "temperature": request.temperature.unwrap_or(0.7),
    })).map_err(|e| e.to_string())?;
    Ok(())
}

/// OpenAI-compatible streaming (Ollama, OpenRouter, Kimi, Minimax, Custom)
async fn stream_openai_compatible(
    app: &AppHandle,
    request_id: &str,
    request: &CompletionRequest,
    client: &Client,
    url: &str,
    api_key: Option<&str>,
) -> Result<(), String> {
    let mut messages: Vec<Value> = vec![];

    // System prompt as first message
    if let Some(system) = &request.system {
        messages.push(json!({ "role": "system", "content": system }));
    }
    for msg in &request.messages {
        messages.push(json!({ "role": msg.role, "content": msg.content }));
    }

    let body = json!({
        "model": request.model,
        "messages": messages,
        "stream": true,
        "max_tokens": request.max_tokens.unwrap_or(1024),
        "temperature": request.temperature.unwrap_or(0.7),
    });

    let mut req = client.post(url)
        .header("Content-Type", "application/json");

    if let Some(key) = api_key {
        req = req.header("Authorization", format!("Bearer {}", key));
    }

    let response = req
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Request failed: {}", e))?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        return Err(format!("Provider error {}: {}", status, text));
    }

    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Stream error: {}", e))?;
        let text = String::from_utf8_lossy(&chunk);

        for line in text.lines() {
            if !line.starts_with("data: ") { continue; }
            let data = &line["data: ".len()..];
            if data == "[DONE]" {
                app.emit("llm-done", StreamDone {
                    request_id: request_id.to_string(),
                }).map_err(|e| e.to_string())?;
                return Ok(());
            }
            if let Ok(json) = serde_json::from_str::<Value>(data) {
                if let Some(delta) = json["choices"][0]["delta"]["content"].as_str() {
                    if !delta.is_empty() {
                        app.emit("llm-delta", StreamDelta {
                            request_id: request_id.to_string(),
                            delta: delta.to_string(),
                        }).map_err(|e| e.to_string())?;
                    }
                }
            }
        }
    }

    app.emit("llm-done", StreamDone {
        request_id: request_id.to_string(),
    }).map_err(|e| e.to_string())?;

    Ok(())
}

/// Anthropic streaming — different API format from OpenAI
async fn stream_anthropic(
    app: &AppHandle,
    request_id: &str,
    request: &CompletionRequest,
    client: &Client,
    api_key: Option<&str>,
) -> Result<(), String> {
    let key = api_key.ok_or("Anthropic API key required")?;

    let messages: Vec<Value> = request.messages.iter()
        .map(|m| json!({ "role": m.role, "content": m.content }))
        .collect();

    let mut body = json!({
        "model": request.model,
        "messages": messages,
        "max_tokens": request.max_tokens.unwrap_or(1024),
        "stream": true,
    });

    if let Some(system) = &request.system {
        body["system"] = json!(system);
    }

    let response = client
        .post("https://api.anthropic.com/v1/messages")
        .header("Content-Type", "application/json")
        .header("x-api-key", key)
        .header("anthropic-version", "2023-06-01")
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Anthropic request failed: {}", e))?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        return Err(format!("Anthropic error {}: {}", status, text));
    }

    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Stream error: {}", e))?;
        let text = String::from_utf8_lossy(&chunk);

        for line in text.lines() {
            if !line.starts_with("data: ") { continue; }
            let data = &line["data: ".len()..];
            if let Ok(json) = serde_json::from_str::<Value>(data) {
                match json["type"].as_str() {
                    Some("content_block_delta") => {
                        if let Some(delta) = json["delta"]["text"].as_str() {
                            if !delta.is_empty() {
                                app.emit("llm-delta", StreamDelta {
                                    request_id: request_id.to_string(),
                                    delta: delta.to_string(),
                                }).map_err(|e| e.to_string())?;
                            }
                        }
                    }
                    Some("message_stop") => {
                        app.emit("llm-done", StreamDone {
                            request_id: request_id.to_string(),
                        }).map_err(|e| e.to_string())?;
                        return Ok(());
                    }
                    _ => {}
                }
            }
        }
    }

    app.emit("llm-done", StreamDone {
        request_id: request_id.to_string(),
    }).map_err(|e| e.to_string())?;

    Ok(())
}

// ─────────────────────────────────────────
// NON-STREAMING (for agents / single calls)
// ─────────────────────────────────────────

/// Single completion without streaming — for Koog agent tool calls
/// Returns full response text synchronously
#[tauri::command]
pub async fn providers_complete(
    state: State<'_, ProvidersState>,
    crypto_state: State<'_, crate::crypto::CryptoState>,
    request: CompletionRequest,
) -> Result<String, String> {
    let use_tor = should_use_tor(&state, &request.provider);
    let tor_proxy = state.tor_proxy.clone();

    let api_key = if let Some(key_name) = &request.api_key_name {
        Some(crate::crypto::crypto_keystore_get(crypto_state, key_name.clone())?)
    } else {
        None
    };

    match request.provider {
        ProviderKind::Leap | ProviderKind::Ollama => {
            let client = build_client(false, &tor_proxy)?;
            let url = match request.provider {
                ProviderKind::Leap => "http://localhost:11435/v1/chat/completions".to_string(), // Leap local endpoint
                _ => "http://localhost:11434/v1/chat/completions".to_string(),
            };
            complete_openai_compatible(&request, &client, &url, None).await
        }
        ProviderKind::Anthropic => {
            let client = build_client(use_tor, &tor_proxy)?;
            complete_anthropic(&request, &client, api_key.as_deref()).await
        }
        ProviderKind::OpenRouter => {
            let client = build_client(use_tor, &tor_proxy)?;
            complete_openai_compatible(&request, &client,
                "https://openrouter.ai/api/v1/chat/completions",
                api_key.as_deref()).await
        }
        ProviderKind::Kimi => {
            let client = build_client(use_tor, &tor_proxy)?;
            complete_openai_compatible(&request, &client,
                "https://api.moonshot.cn/v1/chat/completions",
                api_key.as_deref()).await
        }
        ProviderKind::Minimax => {
            let client = build_client(use_tor, &tor_proxy)?;
            complete_openai_compatible(&request, &client,
                "https://api.minimax.chat/v1/text/chatcompletion_v2",
                api_key.as_deref()).await
        }
        ProviderKind::Custom => {
            let client = build_client(use_tor, &tor_proxy)?;
            let url = request.api_url.clone()
                .ok_or("Custom provider requires api_url")?;
            complete_openai_compatible(&request, &client, &url, api_key.as_deref()).await
        }
    }
}

async fn complete_openai_compatible(
    request: &CompletionRequest,
    client: &Client,
    url: &str,
    api_key: Option<&str>,
) -> Result<String, String> {
    let mut messages: Vec<Value> = vec![];
    if let Some(system) = &request.system {
        messages.push(json!({ "role": "system", "content": system }));
    }
    for msg in &request.messages {
        messages.push(json!({ "role": msg.role, "content": msg.content }));
    }
    let body = json!({
        "model": request.model,
        "messages": messages,
        "max_tokens": request.max_tokens.unwrap_or(1024),
        "temperature": request.temperature.unwrap_or(0.7),
        "stream": false,
    });
    let mut req = client.post(url).header("Content-Type", "application/json");
    if let Some(key) = api_key {
        req = req.header("Authorization", format!("Bearer {}", key));
    }
    let response = req.json(&body).send().await.map_err(|e| e.to_string())?;
    if !response.status().is_success() {
        return Err(format!("Provider error {}: {}", response.status(),
            response.text().await.unwrap_or_default()));
    }
    let json: Value = response.json().await.map_err(|e| e.to_string())?;
    json["choices"][0]["message"]["content"]
        .as_str()
        .map(String::from)
        .ok_or("No content in response".to_string())
}

async fn complete_anthropic(
    request: &CompletionRequest,
    client: &Client,
    api_key: Option<&str>,
) -> Result<String, String> {
    let key = api_key.ok_or("Anthropic API key required")?;
    let messages: Vec<Value> = request.messages.iter()
        .map(|m| json!({ "role": m.role, "content": m.content }))
        .collect();
    let mut body = json!({
        "model": request.model,
        "messages": messages,
        "max_tokens": request.max_tokens.unwrap_or(1024),
    });
    if let Some(system) = &request.system {
        body["system"] = json!(system);
    }
    let response = client
        .post("https://api.anthropic.com/v1/messages")
        .header("Content-Type", "application/json")
        .header("x-api-key", key)
        .header("anthropic-version", "2023-06-01")
        .json(&body)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !response.status().is_success() {
        return Err(format!("Anthropic error {}: {}", response.status(),
            response.text().await.unwrap_or_default()));
    }
    let json: Value = response.json().await.map_err(|e| e.to_string())?;
    json["content"][0]["text"]
        .as_str()
        .map(String::from)
        .ok_or("No content in response".to_string())
}

// ─────────────────────────────────────────
// REGISTER ALL COMMANDS in main.rs:
//
// .invoke_handler(tauri::generate_handler![
//     providers::providers_list,
//     providers::providers_tor_set,
//     providers::providers_tor_status,
//     providers::providers_stream,
//     providers::providers_complete,
// ])
// ─────────────────────────────────────────
