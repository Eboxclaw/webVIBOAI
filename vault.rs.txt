/// vault.rs — ViBo Encrypted Vault
///
/// Encrypted notes stored as .md files with AES-256-GCM content.
/// Structure is identical to notes.rs but all content is encrypted at rest.
///
/// File format on disk:
///   ---
///   id: uuid
///   title: "Encrypted"        ← title is also encrypted, shown as placeholder
///   vault: true               ← marker so UI knows this is a vault note
///   encrypted_title: <base64> ← encrypted real title
///   created: ...
///   modified: ...
///   ---
///   <EncryptedBlob JSON>      ← body is serialised EncryptedBlob
///
/// Vault notes are stored in /vault/ subfolder — hidden from notes UI.
/// Requires vault to be unlocked (crypto_unlock called first).
///
/// Depends on:
///   crypto.rs  → CryptoState, crypto_encrypt_note, crypto_decrypt_note
///   notes.rs   → WikiLink type
///
/// Cargo.toml: same as notes.rs + crypto.rs

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use tauri::State;
use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::crypto::{CryptoState, EncryptedBlob};

// ─────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct VaultNote {
    pub id: String,
    pub title: String,              // decrypted title
    pub content: String,            // decrypted markdown body
    pub created_at: DateTime<Utc>,
    pub modified_at: DateTime<Utc>,
    pub path: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct VaultNoteStub {
    pub id: String,
    pub title: String,              // decrypted title
    pub created_at: DateTime<Utc>,
    pub modified_at: DateTime<Utc>,
}

/// Raw file format stored on disk — all sensitive fields encrypted
#[derive(Debug, Serialize, Deserialize)]
struct VaultNoteRaw {
    id: String,
    encrypted_title: String,        // base64 EncryptedBlob JSON
    encrypted_body: String,         // base64 EncryptedBlob JSON
    created_at: String,
    modified_at: String,
}

pub struct VaultState {
    pub vault_path: PathBuf,
}

// ─────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────

fn vault_dir(vault_path: &Path) -> PathBuf {
    vault_path.join("vault")
}

fn note_path(vault_path: &Path, id: &str) -> PathBuf {
    vault_dir(vault_path).join(format!("{}.md", id))
}

fn ensure_vault_dir(vault_path: &Path) -> Result<(), String> {
    fs::create_dir_all(vault_dir(vault_path)).map_err(|e| e.to_string())
}

fn encrypt_string(
    crypto: &CryptoState,
    plaintext: &str,
) -> Result<String, String> {
    let blob = crate::crypto::crypto_encrypt_note(
        // We call the internal function directly
        // since we're in the same crate
        State::new_for_testing(crypto), // placeholder — see note below
        plaintext.to_string(),
    )?;
    serde_json::to_string(&blob).map_err(|e| e.to_string())
}

fn decrypt_string(
    crypto: &CryptoState,
    blob_json: &str,
) -> Result<String, String> {
    let blob: EncryptedBlob = serde_json::from_str(blob_json)
        .map_err(|e| format!("Invalid encrypted blob: {}", e))?;

    // Decrypt using session key directly
    let session = crypto.session_key.lock().unwrap();
    let key = session.as_ref().ok_or("Vault is locked")?;

    let ciphertext = base64::engine::general_purpose::STANDARD
        .decode(&blob.ciphertext)
        .map_err(|e| e.to_string())?;
    let nonce_bytes = base64::engine::general_purpose::STANDARD
        .decode(&blob.nonce)
        .map_err(|e| e.to_string())?;

    use aes_gcm::{Aes256Gcm, Key, Nonce, aead::{Aead, KeyInit}};
    let aes_key = Key::<Aes256Gcm>::from_slice(key);
    let cipher = Aes256Gcm::new(aes_key);
    let nonce = Nonce::from_slice(&nonce_bytes);

    let plaintext = cipher
        .decrypt(nonce, ciphertext.as_ref())
        .map_err(|_| "Decryption failed — vault may be locked or data corrupted".to_string())?;

    String::from_utf8(plaintext).map_err(|e| e.to_string())
}

fn encrypt_string_raw(crypto: &CryptoState, plaintext: &str) -> Result<String, String> {
    let session = crypto.session_key.lock().unwrap();
    let key = session.as_ref().ok_or("Vault is locked")?;

    use aes_gcm::{Aes256Gcm, Key, Nonce, aead::{Aead, AeadCore, KeyInit, OsRng}};
    use base64::engine::general_purpose::STANDARD as B64;
    use base64::Engine;

    let aes_key = Key::<Aes256Gcm>::from_slice(key);
    let cipher = Aes256Gcm::new(aes_key);
    let nonce = Aes256Gcm::generate_nonce(&mut OsRng);

    let ciphertext = cipher
        .encrypt(&nonce, plaintext.as_bytes())
        .map_err(|e| format!("Encrypt error: {}", e))?;

    let blob = EncryptedBlob {
        ciphertext: B64.encode(&ciphertext),
        nonce: B64.encode(nonce.as_slice()),
        salt: String::new(),
        kdf_version: 1,
    };

    serde_json::to_string(&blob).map_err(|e| e.to_string())
}

fn write_raw_to_disk(vault_path: &Path, raw: &VaultNoteRaw) -> Result<(), String> {
    let path = note_path(vault_path, &raw.id);
    let content = format!(
        "---\nid: {}\nvault: true\ncreated_at: {}\nmodified_at: {}\n---\n\n{}\n\n{}\n",
        raw.id,
        raw.created_at,
        raw.modified_at,
        raw.encrypted_title,
        raw.encrypted_body,
    );
    fs::write(&path, content).map_err(|e| e.to_string())
}

fn read_raw_from_disk(path: &Path) -> Option<VaultNoteRaw> {
    let content = fs::read_to_string(path).ok()?;
    if !content.starts_with("---") { return None; }
    let rest = &content[3..];
    let end = rest.find("\n---")?;
    let yaml = &rest[..end];
    let body = rest[end + 4..].trim_start_matches('\n');

    let val: serde_yaml::Value = serde_yaml::from_str(yaml).ok()?;
    let id = val["id"].as_str()?.to_string();
    let created_at = val["created_at"].as_str().unwrap_or("").to_string();
    let modified_at = val["modified_at"].as_str().unwrap_or("").to_string();

    // Body has two lines: encrypted_title then encrypted_body
    let mut lines = body.lines().filter(|l| !l.is_empty());
    let encrypted_title = lines.next()?.to_string();
    let encrypted_body = lines.next()?.to_string();

    Some(VaultNoteRaw {
        id,
        encrypted_title,
        encrypted_body,
        created_at,
        modified_at,
    })
}

// ─────────────────────────────────────────
// TAURI COMMANDS
// ─────────────────────────────────────────

/// Create a new encrypted note in the vault
/// Vault must be unlocked
#[tauri::command]
pub fn vault_create(
    state: State<VaultState>,
    crypto: State<CryptoState>,
    title: String,
    content: String,
) -> Result<VaultNoteStub, String> {
    ensure_vault_dir(&state.vault_path)?;

    let id = Uuid::new_v4().to_string();
    let now = Utc::now();
    let now_str = now.to_rfc3339();

    let encrypted_title = encrypt_string_raw(&crypto, &title)?;
    let encrypted_body = encrypt_string_raw(&crypto, &content)?;

    let raw = VaultNoteRaw {
        id: id.clone(),
        encrypted_title,
        encrypted_body,
        created_at: now_str.clone(),
        modified_at: now_str.clone(),
    };

    write_raw_to_disk(&state.vault_path, &raw)?;

    Ok(VaultNoteStub {
        id,
        title,
        created_at: now,
        modified_at: now,
    })
}

/// Read and decrypt a vault note
/// Vault must be unlocked
#[tauri::command]
pub fn vault_read(
    state: State<VaultState>,
    crypto: State<CryptoState>,
    id: String,
) -> Result<VaultNote, String> {
    let path = note_path(&state.vault_path, &id);
    let raw = read_raw_from_disk(&path)
        .ok_or_else(|| format!("Vault note not found: {}", id))?;

    let title = decrypt_string(&crypto, &raw.encrypted_title)?;
    let content = decrypt_string(&crypto, &raw.encrypted_body)?;

    let created_at = DateTime::parse_from_rfc3339(&raw.created_at)
        .map(|d| d.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now());
    let modified_at = DateTime::parse_from_rfc3339(&raw.modified_at)
        .map(|d| d.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now());

    Ok(VaultNote {
        id,
        title,
        content,
        created_at,
        modified_at,
        path: path.to_string_lossy().to_string(),
    })
}

/// Update a vault note's content
/// Vault must be unlocked
#[tauri::command]
pub fn vault_write(
    state: State<VaultState>,
    crypto: State<CryptoState>,
    id: String,
    title: Option<String>,
    content: Option<String>,
) -> Result<VaultNoteStub, String> {
    let path = note_path(&state.vault_path, &id);
    let raw = read_raw_from_disk(&path)
        .ok_or_else(|| format!("Vault note not found: {}", id))?;

    // Decrypt existing values to keep unchanged fields
    let existing_title = decrypt_string(&crypto, &raw.encrypted_title)?;
    let existing_content = decrypt_string(&crypto, &raw.encrypted_body)?;

    let new_title = title.unwrap_or(existing_title.clone());
    let new_content = content.unwrap_or(existing_content);
    let now = Utc::now();

    let encrypted_title = encrypt_string_raw(&crypto, &new_title)?;
    let encrypted_body = encrypt_string_raw(&crypto, &new_content)?;

    let updated = VaultNoteRaw {
        id: id.clone(),
        encrypted_title,
        encrypted_body,
        created_at: raw.created_at,
        modified_at: now.to_rfc3339(),
    };

    write_raw_to_disk(&state.vault_path, &updated)?;

    Ok(VaultNoteStub {
        id,
        title: new_title,
        created_at: DateTime::parse_from_rfc3339(&updated.created_at)
            .map(|d| d.with_timezone(&Utc))
            .unwrap_or(now),
        modified_at: now,
    })
}

/// Delete a vault note — moves to .trash (encrypted)
#[tauri::command]
pub fn vault_delete(
    state: State<VaultState>,
    id: String,
) -> Result<(), String> {
    let path = note_path(&state.vault_path, &id);
    if !path.exists() {
        return Err(format!("Vault note not found: {}", id));
    }
    let trash = state.vault_path.join(".trash").join("vault").join(format!("{}.md", id));
    fs::create_dir_all(trash.parent().unwrap()).map_err(|e| e.to_string())?;
    fs::rename(&path, &trash).map_err(|e| e.to_string())
}

/// List all vault notes — decrypts titles only
/// Vault must be unlocked
#[tauri::command]
pub fn vault_list(
    state: State<VaultState>,
    crypto: State<CryptoState>,
) -> Result<Vec<VaultNoteStub>, String> {
    let vault_dir = vault_dir(&state.vault_path);
    if !vault_dir.exists() { return Ok(vec![]); }

    let mut stubs = vec![];
    for entry in fs::read_dir(&vault_dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if path.extension().map(|e| e == "md").unwrap_or(false) {
            if let Some(raw) = read_raw_from_disk(&path) {
                // If vault is locked, return placeholder titles
                let title = decrypt_string(&crypto, &raw.encrypted_title)
                    .unwrap_or_else(|_| "🔒 Locked".to_string());

                let created_at = DateTime::parse_from_rfc3339(&raw.created_at)
                    .map(|d| d.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now());
                let modified_at = DateTime::parse_from_rfc3339(&raw.modified_at)
                    .map(|d| d.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now());

                stubs.push(VaultNoteStub {
                    id: raw.id,
                    title,
                    created_at,
                    modified_at,
                });
            }
        }
    }

    stubs.sort_by(|a, b| b.modified_at.cmp(&a.modified_at));
    Ok(stubs)
}

/// Search vault notes — decrypts all and searches in memory
/// Only possible when vault is unlocked
#[tauri::command]
pub fn vault_search(
    state: State<VaultState>,
    crypto: State<CryptoState>,
    query: String,
) -> Result<Vec<VaultNoteStub>, String> {
    let q = query.to_lowercase();
    let all = vault_list(state.clone(), crypto.clone())?;

    // For stubs we only have title — for full search we need to read each note
    // This is intentionally slower to avoid leaking unencrypted content to SQLite
    let vault_dir = vault_dir(&state.vault_path);
    let mut results = vec![];

    for stub in all {
        if stub.title.to_lowercase().contains(&q) {
            results.push(stub);
            continue;
        }
        // Check body
        let path = vault_dir.join(format!("{}.md", stub.id));
        if let Some(raw) = read_raw_from_disk(&path) {
            if let Ok(content) = decrypt_string(&crypto, &raw.encrypted_body) {
                if content.to_lowercase().contains(&q) {
                    results.push(stub);
                }
            }
        }
    }

    Ok(results)
}

/// Snapshot a vault note (encrypted snapshot)
#[tauri::command]
pub fn vault_snapshot(
    state: State<VaultState>,
    id: String,
) -> Result<String, String> {
    let path = note_path(&state.vault_path, &id);
    let content = fs::read_to_string(&path).map_err(|e| e.to_string())?;
    let ts = Utc::now().format("%Y%m%d%H%M%S").to_string();
    let snapshot_path = state.vault_path
        .join(".snapshots")
        .join("vault")
        .join(&id)
        .join(format!("{}.md", ts));

    fs::create_dir_all(snapshot_path.parent().unwrap()).map_err(|e| e.to_string())?;
    fs::write(&snapshot_path, content).map_err(|e| e.to_string())?;

    Ok(format!(".snapshots/vault/{}/{}.md", id, ts))
}

/// Restore vault note from snapshot
#[tauri::command]
pub fn vault_restore(
    state: State<VaultState>,
    id: String,
    snapshot_id: String,
) -> Result<(), String> {
    let snapshot_path = state.vault_path.join(&snapshot_id);
    let content = fs::read_to_string(&snapshot_path).map_err(|e| e.to_string())?;
    let path = note_path(&state.vault_path, &id);
    fs::write(&path, content).map_err(|e| e.to_string())
}

/// Count vault notes — works even when locked (no decryption needed)
#[tauri::command]
pub fn vault_count(state: State<VaultState>) -> Result<usize, String> {
    let vault_dir = vault_dir(&state.vault_path);
    if !vault_dir.exists() { return Ok(0); }
    let count = fs::read_dir(&vault_dir)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "md").unwrap_or(false))
        .count();
    Ok(count)
}

// ─────────────────────────────────────────
// REGISTER ALL COMMANDS in main.rs:
//
// .invoke_handler(tauri::generate_handler![
//     vault::vault_create,
//     vault::vault_read,
//     vault::vault_write,
//     vault::vault_delete,
//     vault::vault_list,
//     vault::vault_search,
//     vault::vault_snapshot,
//     vault::vault_restore,
//     vault::vault_count,
// ])
// ─────────────────────────────────────────
