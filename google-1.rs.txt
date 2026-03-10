/// google.rs — ViBo Google API
///
/// Calendar (read + write) and Gmail (read only).
/// All OAuth handled by oauth.rs — this module only makes API calls.
///
/// Token retrieval: oauth_get_token("google") → auto-refreshes if needed
/// Tor: respects ProvidersState tor_enabled
///
/// Depends on:
///   oauth.rs     → token management
///   providers.rs → Tor toggle

use chrono::Utc;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::State;

use crate::oauth::OAuthState;
use crate::providers::ProvidersState;

const CALENDAR_BASE: &str = "https://www.googleapis.com/calendar/v3";
const GMAIL_BASE: &str    = "https://gmail.googleapis.com/gmail/v1";

// ─────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CalendarEvent {
    pub id: String,
    pub title: String,
    pub description: Option<String>,
    pub start: String,
    pub end: String,
    pub all_day: bool,
    pub calendar_id: String,
    pub html_link: Option<String>,
    pub status: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CalendarList {
    pub id: String,
    pub name: String,
    pub primary: bool,
    pub color: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateEventArgs {
    pub calendar_id: Option<String>,
    pub title: String,
    pub description: Option<String>,
    pub start: String,
    pub end: String,
    pub all_day: Option<bool>,
    pub timezone: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UpdateEventArgs {
    pub calendar_id: Option<String>,
    pub event_id: String,
    pub title: Option<String>,
    pub description: Option<String>,
    pub start: Option<String>,
    pub end: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GmailMessage {
    pub id: String,
    pub thread_id: String,
    pub subject: Option<String>,
    pub from: Option<String>,
    pub to: Option<String>,
    pub snippet: String,
    pub date: Option<String>,
    pub unread: bool,
    pub labels: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GmailMessageFull {
    pub id: String,
    pub thread_id: String,
    pub subject: Option<String>,
    pub from: Option<String>,
    pub to: Option<String>,
    pub date: Option<String>,
    pub body_plain: Option<String>,
    pub body_html: Option<String>,
    pub labels: Vec<String>,
}

// ─────────────────────────────────────────
// HTTP CLIENT
// ─────────────────────────────────────────

fn build_client(providers: &ProvidersState) -> Result<Client, String> {
    let tor_on = *providers.tor_enabled.lock().unwrap();
    let mut builder = Client::builder()
        .timeout(std::time::Duration::from_secs(30));
    if tor_on {
        let proxy = reqwest::Proxy::all(&providers.tor_proxy)
            .map_err(|e| e.to_string())?;
        builder = builder.proxy(proxy);
    }
    builder.build().map_err(|e| e.to_string())
}

// ─────────────────────────────────────────
// CALENDAR COMMANDS
// ─────────────────────────────────────────

#[tauri::command]
pub async fn google_calendar_list(
    oauth: State<'_, OAuthState>,
    crypto: State<'_, crate::crypto::CryptoState>,
    providers: State<'_, ProvidersState>,
) -> Result<Vec<CalendarList>, String> {
    let client = build_client(&providers)?;
    let token = crate::oauth::oauth_get_token(oauth, crypto, providers, "google".to_string()).await?;

    let json: Value = client
        .get(format!("{}/users/me/calendarList", CALENDAR_BASE))
        .bearer_auth(&token)
        .send().await.map_err(|e| e.to_string())?
        .json().await.map_err(|e| e.to_string())?;

    Ok(json["items"].as_array().unwrap_or(&vec![]).iter().map(|c| CalendarList {
        id: c["id"].as_str().unwrap_or("").to_string(),
        name: c["summary"].as_str().unwrap_or("").to_string(),
        primary: c["primary"].as_bool().unwrap_or(false),
        color: c["backgroundColor"].as_str().map(String::from),
    }).collect())
}

#[tauri::command]
pub async fn google_calendar_events(
    oauth: State<'_, OAuthState>,
    crypto: State<'_, crate::crypto::CryptoState>,
    providers: State<'_, ProvidersState>,
    calendar_id: Option<String>,
    date_from: String,
    date_to: String,
    max_results: Option<u32>,
) -> Result<Vec<CalendarEvent>, String> {
    let client = build_client(&providers)?;
    let token = crate::oauth::oauth_get_token(oauth, crypto, providers, "google".to_string()).await?;
    let cal = calendar_id.unwrap_or_else(|| "primary".to_string());
    let time_min = normalise_datetime(&date_from, false);
    let time_max = normalise_datetime(&date_to, true);

    let json: Value = client
        .get(format!("{}/calendars/{}/events", CALENDAR_BASE, cal))
        .bearer_auth(&token)
        .query(&[
            ("timeMin",      time_min.as_str()),
            ("timeMax",      time_max.as_str()),
            ("maxResults",   &max_results.unwrap_or(50).to_string()),
            ("singleEvents", "true"),
            ("orderBy",      "startTime"),
        ])
        .send().await.map_err(|e| e.to_string())?
        .json().await.map_err(|e| e.to_string())?;

    Ok(parse_events(&json, &cal))
}

#[tauri::command]
pub async fn google_calendar_today(
    oauth: State<'_, OAuthState>,
    crypto: State<'_, crate::crypto::CryptoState>,
    providers: State<'_, ProvidersState>,
    calendar_id: Option<String>,
) -> Result<Vec<CalendarEvent>, String> {
    let today    = chrono::Local::now().date_naive();
    let tomorrow = today + chrono::Duration::days(1);
    google_calendar_events(
        oauth, crypto, providers, calendar_id,
        today.format("%Y-%m-%d").to_string(),
        tomorrow.format("%Y-%m-%d").to_string(),
        None,
    ).await
}

#[tauri::command]
pub async fn google_calendar_create(
    oauth: State<'_, OAuthState>,
    crypto: State<'_, crate::crypto::CryptoState>,
    providers: State<'_, ProvidersState>,
    args: CreateEventArgs,
) -> Result<CalendarEvent, String> {
    let client = build_client(&providers)?;
    let token = crate::oauth::oauth_get_token(oauth, crypto, providers, "google".to_string()).await?;
    let cal = args.calendar_id.clone().unwrap_or_else(|| "primary".to_string());
    let tz = args.timezone.clone().unwrap_or_else(|| "UTC".to_string());
    let all_day = args.all_day.unwrap_or(false);

    let (start_val, end_val) = if all_day {
        (json!({ "date": args.start }), json!({ "date": args.end }))
    } else {
        (json!({ "dateTime": args.start, "timeZone": tz }),
         json!({ "dateTime": args.end,   "timeZone": tz }))
    };

    let mut body = json!({ "summary": args.title, "start": start_val, "end": end_val });
    if let Some(desc) = &args.description { body["description"] = json!(desc); }

    let response = client
        .post(format!("{}/calendars/{}/events", CALENDAR_BASE, cal))
        .bearer_auth(&token)
        .json(&body)
        .send().await.map_err(|e| e.to_string())?;

    if !response.status().is_success() {
        return Err(format!("Calendar create error {}: {}",
            response.status(), response.text().await.unwrap_or_default()));
    }

    let json: Value = response.json().await.map_err(|e| e.to_string())?;
    parse_single_event(&json, &cal).ok_or("Failed to parse created event".to_string())
}

#[tauri::command]
pub async fn google_calendar_update(
    oauth: State<'_, OAuthState>,
    crypto: State<'_, crate::crypto::CryptoState>,
    providers: State<'_, ProvidersState>,
    args: UpdateEventArgs,
) -> Result<CalendarEvent, String> {
    let client = build_client(&providers)?;
    let token = crate::oauth::oauth_get_token(oauth, crypto, providers.clone(), "google".to_string()).await?;
    let cal = args.calendar_id.clone().unwrap_or_else(|| "primary".to_string());

    let mut body: Value = client
        .get(format!("{}/calendars/{}/events/{}", CALENDAR_BASE, cal, args.event_id))
        .bearer_auth(&token)
        .send().await.map_err(|e| e.to_string())?
        .json().await.map_err(|e| e.to_string())?;

    if let Some(t) = &args.title       { body["summary"] = json!(t); }
    if let Some(d) = &args.description { body["description"] = json!(d); }
    if let Some(s) = &args.start       { body["start"] = json!({ "dateTime": s, "timeZone": "UTC" }); }
    if let Some(e) = &args.end         { body["end"]   = json!({ "dateTime": e, "timeZone": "UTC" }); }

    let response = client
        .put(format!("{}/calendars/{}/events/{}", CALENDAR_BASE, cal, args.event_id))
        .bearer_auth(&token)
        .json(&body)
        .send().await.map_err(|e| e.to_string())?;

    if !response.status().is_success() {
        return Err(format!("Calendar update error {}: {}",
            response.status(), response.text().await.unwrap_or_default()));
    }

    let json: Value = response.json().await.map_err(|e| e.to_string())?;
    parse_single_event(&json, &cal).ok_or("Failed to parse updated event".to_string())
}

#[tauri::command]
pub async fn google_calendar_delete(
    oauth: State<'_, OAuthState>,
    crypto: State<'_, crate::crypto::CryptoState>,
    providers: State<'_, ProvidersState>,
    calendar_id: Option<String>,
    event_id: String,
) -> Result<(), String> {
    let client = build_client(&providers)?;
    let token = crate::oauth::oauth_get_token(oauth, crypto, providers, "google".to_string()).await?;
    let cal = calendar_id.unwrap_or_else(|| "primary".to_string());

    let response = client
        .delete(format!("{}/calendars/{}/events/{}", CALENDAR_BASE, cal, event_id))
        .bearer_auth(&token)
        .send().await.map_err(|e| e.to_string())?;

    if !response.status().is_success() && response.status().as_u16() != 204 {
        return Err(format!("Calendar delete error: {}", response.status()));
    }
    Ok(())
}

// ─────────────────────────────────────────
// GMAIL COMMANDS (read only)
// ─────────────────────────────────────────

#[tauri::command]
pub async fn google_gmail_list(
    oauth: State<'_, OAuthState>,
    crypto: State<'_, crate::crypto::CryptoState>,
    providers: State<'_, ProvidersState>,
    query: Option<String>,
    max_results: Option<u32>,
) -> Result<Vec<GmailMessage>, String> {
    let client = build_client(&providers)?;
    let token = crate::oauth::oauth_get_token(oauth, crypto, providers, "google".to_string()).await?;

    let mut params = vec![("maxResults", max_results.unwrap_or(20).to_string())];
    if let Some(q) = &query { params.push(("q", q.clone())); }

    let list: Value = client
        .get(format!("{}/users/me/messages", GMAIL_BASE))
        .bearer_auth(&token)
        .query(&params)
        .send().await.map_err(|e| e.to_string())?
        .json().await.map_err(|e| e.to_string())?;

    let ids: Vec<String> = list["messages"].as_array().unwrap_or(&vec![])
        .iter().filter_map(|m| m["id"].as_str().map(String::from)).collect();

    let mut messages = vec![];
    for id in ids {
        let msg: Value = client
            .get(format!("{}/users/me/messages/{}", GMAIL_BASE, id))
            .bearer_auth(&token)
            .query(&[("format", "metadata"), ("metadataHeaders", "Subject,From,To,Date")])
            .send().await.map_err(|e| e.to_string())?
            .json().await.map_err(|e| e.to_string())?;
        if let Some(parsed) = parse_gmail_message(&msg) {
            messages.push(parsed);
        }
    }
    Ok(messages)
}

#[tauri::command]
pub async fn google_gmail_read(
    oauth: State<'_, OAuthState>,
    crypto: State<'_, crate::crypto::CryptoState>,
    providers: State<'_, ProvidersState>,
    message_id: String,
) -> Result<GmailMessageFull, String> {
    let client = build_client(&providers)?;
    let token = crate::oauth::oauth_get_token(oauth, crypto, providers, "google".to_string()).await?;

    let msg: Value = client
        .get(format!("{}/users/me/messages/{}", GMAIL_BASE, message_id))
        .bearer_auth(&token)
        .query(&[("format", "full")])
        .send().await.map_err(|e| e.to_string())?
        .json().await.map_err(|e| e.to_string())?;

    parse_gmail_message_full(&msg).ok_or("Failed to parse Gmail message".to_string())
}

#[tauri::command]
pub async fn google_gmail_unread_count(
    oauth: State<'_, OAuthState>,
    crypto: State<'_, crate::crypto::CryptoState>,
    providers: State<'_, ProvidersState>,
) -> Result<u64, String> {
    let client = build_client(&providers)?;
    let token = crate::oauth::oauth_get_token(oauth, crypto, providers, "google".to_string()).await?;

    let resp: Value = client
        .get(format!("{}/users/me/messages", GMAIL_BASE))
        .bearer_auth(&token)
        .query(&[("q", "is:unread"), ("maxResults", "1")])
        .send().await.map_err(|e| e.to_string())?
        .json().await.map_err(|e| e.to_string())?;

    Ok(resp["resultSizeEstimate"].as_u64().unwrap_or(0))
}

// ─────────────────────────────────────────
// PARSE HELPERS
// ─────────────────────────────────────────

fn parse_events(json: &Value, calendar_id: &str) -> Vec<CalendarEvent> {
    json["items"].as_array().unwrap_or(&vec![])
        .iter().filter_map(|e| parse_single_event(e, calendar_id)).collect()
}

fn parse_single_event(e: &Value, calendar_id: &str) -> Option<CalendarEvent> {
    let id = e["id"].as_str()?.to_string();
    let title = e["summary"].as_str().unwrap_or("(No title)").to_string();
    let (start, all_day) = if let Some(d) = e["start"]["date"].as_str() {
        (d.to_string(), true)
    } else {
        (e["start"]["dateTime"].as_str().unwrap_or("").to_string(), false)
    };
    let end = if all_day {
        e["end"]["date"].as_str().unwrap_or("").to_string()
    } else {
        e["end"]["dateTime"].as_str().unwrap_or("").to_string()
    };
    Some(CalendarEvent {
        id, title,
        description: e["description"].as_str().map(String::from),
        start, end, all_day,
        calendar_id: calendar_id.to_string(),
        html_link: e["htmlLink"].as_str().map(String::from),
        status: e["status"].as_str().map(String::from),
    })
}

fn parse_gmail_message(msg: &Value) -> Option<GmailMessage> {
    let id        = msg["id"].as_str()?.to_string();
    let thread_id = msg["threadId"].as_str().unwrap_or("").to_string();
    let snippet   = msg["snippet"].as_str().unwrap_or("").to_string();
    let labels: Vec<String> = msg["labelIds"].as_array().unwrap_or(&vec![])
        .iter().filter_map(|l| l.as_str().map(String::from)).collect();
    let unread = labels.contains(&"UNREAD".to_string());
    let headers = msg["payload"]["headers"].as_array();
    let get_header = |name: &str| -> Option<String> {
        headers?.iter().find(|h| h["name"].as_str() == Some(name))
            .and_then(|h| h["value"].as_str().map(String::from))
    };
    Some(GmailMessage {
        id, thread_id,
        subject: get_header("Subject"),
        from: get_header("From"),
        to: get_header("To"),
        snippet, date: get_header("Date"),
        unread, labels,
    })
}

fn parse_gmail_message_full(msg: &Value) -> Option<GmailMessageFull> {
    let base = parse_gmail_message(msg)?;
    let (body_plain, body_html) = extract_body(msg);
    Some(GmailMessageFull {
        id: base.id, thread_id: base.thread_id,
        subject: base.subject, from: base.from, to: base.to,
        date: base.date, body_plain, body_html, labels: base.labels,
    })
}

fn extract_body(msg: &Value) -> (Option<String>, Option<String>) {
    let mut plain = None;
    let mut html  = None;
    extract_parts(&msg["payload"], &mut plain, &mut html);
    (plain, html)
}

fn extract_parts(part: &Value, plain: &mut Option<String>, html: &mut Option<String>) {
    let mime = part["mimeType"].as_str().unwrap_or("");
    if mime == "text/plain" || mime == "text/html" {
        if let Some(data) = part["body"]["data"].as_str() {
            let decoded = base64::engine::general_purpose::URL_SAFE_NO_PAD
                .decode(data).ok()
                .and_then(|b| String::from_utf8(b).ok());
            if mime == "text/plain" { *plain = decoded; }
            else { *html = decoded; }
        }
    }
    if let Some(parts) = part["parts"].as_array() {
        for p in parts { extract_parts(p, plain, html); }
    }
}

fn normalise_datetime(input: &str, end_of_day: bool) -> String {
    if input.contains('T') { return input.to_string(); }
    if end_of_day { format!("{}T23:59:59Z", input) }
    else          { format!("{}T00:00:00Z", input) }
}

// ─────────────────────────────────────────
// REGISTER in main.rs:
//
// .invoke_handler(tauri::generate_handler![
//     google::google_calendar_list,
//     google::google_calendar_events,
//     google::google_calendar_today,
//     google::google_calendar_create,
//     google::google_calendar_update,
//     google::google_calendar_delete,
//     google::google_gmail_list,
//     google::google_gmail_read,
//     google::google_gmail_unread_count,
// ])
// ─────────────────────────────────────────
