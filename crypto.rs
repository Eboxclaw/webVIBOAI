/// crypto.rs — ViBo Encryption Layer
///
/// Responsibilities:
///   - AES-256-GCM encryption/decryption of vault notes + keystore entries
///   - Argon2id key derivation from PIN
///   - Keystore: encrypted storage of API keys + OAuth tokens in SQLite
///   - Biometric gate: delegates auth to tauri-plugin-biometric, then releases key
///
/// What this module does NOT do:
///   - Does not store vault notes (notes.rs)
///   - Does not store embeddings or routing (storage.rs)
///   - Does not call biometric directly from Rust — biometric is called
///     from the frontend via @tauri-apps/plugin-biometric, then crypto
///     commands are called only after auth succeeds
///
/// Cargo.toml dependencies needed:
///   aes-gcm = "0.10"
///   argon2 = "0.5"
///   rand = "0.8"
///   base64 = "0.22"
///   rusqlite = { version = "0.31", features = ["bundled"] }
///   serde = { version = "1", features = ["derive"] }
///   zeroize = "1"
///
/// Setup in lib.rs:
///   .plugin(tauri_plugin_biometric::Builder::new().build())  // mobile only
///   .manage(crypto::CryptoState::new(&vault_path))

use aes_gcm::{
    aead::{Aead, AeadCore, KeyInit, OsRng},
    Aes256Gcm, Key, Nonce,
};
use argon2::{Argon2, PasswordHasher, password_hash::SaltString};
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use rand::RngCore;
use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use zeroize::Zeroize;

// ─────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────

/// Encrypted blob — self-contained, can be stored anywhere (file, SQLite)
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct EncryptedBlob {
    /// base64-encoded ciphertext + AES-GCM tag
    pub ciphertext: String,
    /// base64-encoded 12-byte nonce
    pub nonce: String,
    /// base64-encoded 16-byte Argon2 salt (used for key derivation)
    pub salt: String,
    /// Argon2 params version (for future migration)
    pub kdf_version: u8,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct KeystoreEntry {
    pub key_name: String,
    pub category: String,       // "provider_api_key" | "oauth_token" | "custom"
    pub created_at: String,
    pub modified_at: String,
}

/// Vault lock state — in-memory, never persisted
#[derive(Debug, Clone, PartialEq)]
pub enum VaultState {
    Locked,
    Unlocked,
}

pub struct CryptoState {
    pub db: Mutex<Connection>,
    pub vault_path: PathBuf,
    /// Derived key held in memory while vault is unlocked — zeroed on lock
    session_key: Mutex<Option<Vec<u8>>>,
    vault_state: Mutex<VaultState>,
}

impl CryptoState {
    pub fn new(vault_path: &Path) -> rusqlite::Result<Self> {
        let db_path = vault_path.join(".vibo").join("keystore.db");
        std::fs::create_dir_all(db_path.parent().unwrap())
            .map_err(|e| rusqlite::Error::InvalidPath(e.to_string().into()))?;

        let conn = Connection::open(&db_path)?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             CREATE TABLE IF NOT EXISTS keystore (
                 key_name    TEXT PRIMARY KEY,
                 category    TEXT NOT NULL DEFAULT 'custom',
                 ciphertext  TEXT NOT NULL,
                 nonce       TEXT NOT NULL,
                 salt        TEXT NOT NULL,
                 kdf_version INTEGER NOT NULL DEFAULT 1,
                 created_at  TEXT NOT NULL,
                 modified_at TEXT NOT NULL
             );",
        )?;

        Ok(CryptoState {
            db: Mutex::new(conn),
            vault_path: vault_path.to_path_buf(),
            session_key: Mutex::new(None),
            vault_state: Mutex::new(VaultState::Locked),
        })
    }
}

// ─────────────────────────────────────────
// INTERNAL CRYPTO HELPERS
// ─────────────────────────────────────────

/// Derive 32-byte AES key from PIN using Argon2id
/// salt must be 16 random bytes, stored with the encrypted data
fn derive_key(pin: &str, salt: &[u8]) -> Result<Vec<u8>, String> {
    let argon2 = Argon2::default();
    let salt_b64 = B64.encode(salt);
    let salt_str = SaltString::from_b64(&salt_b64)
        .map_err(|e| format!("Salt error: {}", e))?;

    // Hash is base64url — we use raw bytes of first 32 bytes for AES key
    let hash = argon2
        .hash_password(pin.as_bytes(), &salt_str)
        .map_err(|e| format!("Argon2 error: {}", e))?;

    let hash_bytes = hash
        .hash
        .ok_or("No hash output")?;

    Ok(hash_bytes.as_bytes()[..32].to_vec())
}

fn encrypt_with_key(plaintext: &[u8], key_bytes: &[u8]) -> Result<EncryptedBlob, String> {
    let key = Key::<Aes256Gcm>::from_slice(key_bytes);
    let cipher = Aes256Gcm::new(key);
    let nonce = Aes256Gcm::generate_nonce(&mut OsRng);

    let ciphertext = cipher
        .encrypt(&nonce, plaintext)
        .map_err(|e| format!("Encrypt error: {}", e))?;

    Ok(EncryptedBlob {
        ciphertext: B64.encode(&ciphertext),
        nonce: B64.encode(nonce.as_slice()),
        salt: String::new(), // salt managed separately for session key
        kdf_version: 1,
    })
}

fn decrypt_with_key(blob: &EncryptedBlob, key_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let key = Key::<Aes256Gcm>::from_slice(key_bytes);
    let cipher = Aes256Gcm::new(key);

    let nonce_bytes = B64.decode(&blob.nonce).map_err(|e| e.to_string())?;
    let nonce = Nonce::from_slice(&nonce_bytes);
    let ciphertext = B64.decode(&blob.ciphertext).map_err(|e| e.to_string())?;

    cipher
        .decrypt(nonce, ciphertext.as_ref())
        .map_err(|_| "Decryption failed — wrong key or corrupted data".to_string())
}

fn random_salt() -> [u8; 16] {
    let mut salt = [0u8; 16];
    OsRng.fill_bytes(&mut salt);
    salt
}

// ─────────────────────────────────────────
// VAULT LOCK / UNLOCK COMMANDS
// ─────────────────────────────────────────

/// Unlock vault with PIN — derives session key, holds in memory
/// Call this AFTER biometric auth succeeds on the frontend
#[tauri::command]
pub fn crypto_unlock(
    state: tauri::State<CryptoState>,
    pin: String,
) -> Result<bool, String> {
    // Load the vault salt (stored at first setup)
    let salt = load_vault_salt(&state.vault_path)?;
    let mut key = derive_key(&pin, &salt)?;

    // Verify the key is correct by trying to decrypt the vault verification blob
    if vault_verification_exists(&state.vault_path) {
        let blob = load_vault_verification(&state.vault_path)?;
        decrypt_with_key(&blob, &key)
            .map_err(|_| "Wrong PIN".to_string())?;
    }

    let mut session = state.session_key.lock().unwrap();
    *session = Some(key.clone());
    key.zeroize();

    let mut vs = state.vault_state.lock().unwrap();
    *vs = VaultState::Unlocked;

    Ok(true)
}

/// Lock vault — zeroes session key from memory
#[tauri::command]
pub fn crypto_lock(state: tauri::State<CryptoState>) -> Result<(), String> {
    let mut session = state.session_key.lock().unwrap();
    if let Some(mut k) = session.take() {
        k.zeroize();
    }
    let mut vs = state.vault_state.lock().unwrap();
    *vs = VaultState::Locked;
    Ok(())
}

/// Check if vault is currently unlocked
#[tauri::command]
pub fn crypto_is_unlocked(state: tauri::State<CryptoState>) -> bool {
    let vs = state.vault_state.lock().unwrap();
    *vs == VaultState::Unlocked
}

/// Initial vault setup — set PIN for the first time
/// Creates salt + verification blob so we can validate PIN on unlock
#[tauri::command]
pub fn crypto_setup_vault(
    state: tauri::State<CryptoState>,
    pin: String,
) -> Result<(), String> {
    let salt = random_salt();
    let key = derive_key(&pin, &salt)?;

    // Store salt
    save_vault_salt(&state.vault_path, &salt)?;

    // Store verification blob (encrypts a known constant)
    let blob = encrypt_with_key(b"vibo_vault_ok", &key)?;
    save_vault_verification(&state.vault_path, &blob)?;

    Ok(())
}

/// Change PIN — re-encrypts all keystore entries with new key
#[tauri::command]
pub fn crypto_change_pin(
    state: tauri::State<CryptoState>,
    old_pin: String,
    new_pin: String,
) -> Result<(), String> {
    // Verify old PIN first
    let salt = load_vault_salt(&state.vault_path)?;
    let old_key = derive_key(&old_pin, &salt)?;
    let blob = load_vault_verification(&state.vault_path)?;
    decrypt_with_key(&blob, &old_key)
        .map_err(|_| "Wrong current PIN".to_string())?;

    // New salt + new key
    let new_salt = random_salt();
    let new_key = derive_key(&new_pin, &new_salt)?;

    // Re-encrypt all keystore entries
    let db = state.db.lock().unwrap();
    let entries: Vec<(String, EncryptedBlob)> = {
        let mut stmt = db.prepare(
            "SELECT key_name, ciphertext, nonce, salt, kdf_version FROM keystore"
        ).map_err(|e| e.to_string())?;
        stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                EncryptedBlob {
                    ciphertext: row.get(1)?,
                    nonce: row.get(2)?,
                    salt: row.get(3)?,
                    kdf_version: row.get(4)?,
                },
            ))
        }).map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect()
    };

    for (key_name, old_blob) in entries {
        let plaintext = decrypt_with_key(&old_blob, &old_key)?;
        let new_blob = encrypt_with_key(&plaintext, &new_key)?;
        let now = chrono::Utc::now().to_rfc3339();
        db.execute(
            "UPDATE keystore SET ciphertext = ?1, nonce = ?2, modified_at = ?3 WHERE key_name = ?4",
            params![new_blob.ciphertext, new_blob.nonce, now, key_name],
        ).map_err(|e| e.to_string())?;
    }

    // Update salt + verification blob
    drop(db);
    save_vault_salt(&state.vault_path, &new_salt)?;
    let new_verification = encrypt_with_key(b"vibo_vault_ok", &new_key)?;
    save_vault_verification(&state.vault_path, &new_verification)?;

    // Update session key if currently unlocked
    let mut session = state.session_key.lock().unwrap();
    if session.is_some() {
        *session = Some(new_key);
    }

    Ok(())
}

// ─────────────────────────────────────────
// NOTE ENCRYPTION COMMANDS
// ─────────────────────────────────────────

/// Encrypt a note's content — returns EncryptedBlob
/// Vault must be unlocked. Encrypted .md is saved by vault.rs.
#[tauri::command]
pub fn crypto_encrypt_note(
    state: tauri::State<CryptoState>,
    plaintext: String,
) -> Result<EncryptedBlob, String> {
    let session = state.session_key.lock().unwrap();
    let key = session.as_ref().ok_or("Vault is locked")?;
    let mut blob = encrypt_with_key(plaintext.as_bytes(), key)?;
    blob.salt = B64.encode(load_vault_salt(&state.vault_path)?);
    Ok(blob)
}

/// Decrypt a note's content — vault must be unlocked
#[tauri::command]
pub fn crypto_decrypt_note(
    state: tauri::State<CryptoState>,
    blob: EncryptedBlob,
) -> Result<String, String> {
    let session = state.session_key.lock().unwrap();
    let key = session.as_ref().ok_or("Vault is locked")?;
    let plaintext = decrypt_with_key(&blob, key)?;
    String::from_utf8(plaintext).map_err(|e| e.to_string())
}

// ─────────────────────────────────────────
// KEYSTORE COMMANDS (API keys + OAuth tokens)
// ─────────────────────────────────────────

/// Store an API key or OAuth token — encrypted with vault session key
/// Vault must be unlocked.
#[tauri::command]
pub fn crypto_keystore_set(
    state: tauri::State<CryptoState>,
    key_name: String,           // e.g. "anthropic_api_key", "google_oauth_token"
    secret: String,
    category: Option<String>,   // "provider_api_key" | "oauth_token" | "custom"
) -> Result<(), String> {
    let session = state.session_key.lock().unwrap();
    let key = session.as_ref().ok_or("Vault is locked — unlock before storing keys")?;
    let blob = encrypt_with_key(secret.as_bytes(), key)?;
    let now = chrono::Utc::now().to_rfc3339();
    let cat = category.unwrap_or_else(|| "custom".to_string());
    let db = state.db.lock().unwrap();
    db.execute(
        "INSERT OR REPLACE INTO keystore (key_name, category, ciphertext, nonce, salt, kdf_version, created_at, modified_at)
         VALUES (?1, ?2, ?3, ?4, ?5, 1,
             COALESCE((SELECT created_at FROM keystore WHERE key_name = ?1), ?6), ?6)",
        params![key_name, cat, blob.ciphertext, blob.nonce, blob.salt, now],
    ).map_err(|e| e.to_string())?;
    Ok(())
}

/// Retrieve a secret from keystore — vault must be unlocked
#[tauri::command]
pub fn crypto_keystore_get(
    state: tauri::State<CryptoState>,
    key_name: String,
) -> Result<String, String> {
    let session = state.session_key.lock().unwrap();
    let key = session.as_ref().ok_or("Vault is locked")?;
    let db = state.db.lock().unwrap();
    let blob: EncryptedBlob = db.query_row(
        "SELECT ciphertext, nonce, salt, kdf_version FROM keystore WHERE key_name = ?1",
        params![key_name],
        |row| Ok(EncryptedBlob {
            ciphertext: row.get(0)?,
            nonce: row.get(1)?,
            salt: row.get(2)?,
            kdf_version: row.get(3)?,
        }),
    ).map_err(|_| format!("Key not found: {}", key_name))?;

    let plaintext = decrypt_with_key(&blob, key)?;
    String::from_utf8(plaintext).map_err(|e| e.to_string())
}

/// Delete a keystore entry
#[tauri::command]
pub fn crypto_keystore_delete(
    state: tauri::State<CryptoState>,
    key_name: String,
) -> Result<(), String> {
    let db = state.db.lock().unwrap();
    db.execute("DELETE FROM keystore WHERE key_name = ?1", params![key_name])
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// List keystore entries (names + categories only, no secrets)
#[tauri::command]
pub fn crypto_keystore_list(
    state: tauri::State<CryptoState>,
) -> Result<Vec<KeystoreEntry>, String> {
    let db = state.db.lock().unwrap();
    let mut stmt = db.prepare(
        "SELECT key_name, category, created_at, modified_at FROM keystore ORDER BY category, key_name"
    ).map_err(|e| e.to_string())?;
    let rows = stmt.query_map([], |row| {
        Ok(KeystoreEntry {
            key_name: row.get(0)?,
            category: row.get(1)?,
            created_at: row.get(2)?,
            modified_at: row.get(3)?,
        })
    }).map_err(|e| e.to_string())?;
    rows.map(|r| r.map_err(|e| e.to_string())).collect()
}

// ─────────────────────────────────────────
// VAULT SALT + VERIFICATION HELPERS
// ─────────────────────────────────────────

fn vault_meta_path(vault_path: &Path) -> PathBuf {
    vault_path.join(".vibo").join("vault.meta")
}

fn vault_verification_exists(vault_path: &Path) -> bool {
    vault_path.join(".vibo").join("vault.verify").exists()
}

fn save_vault_salt(vault_path: &Path, salt: &[u8]) -> Result<(), String> {
    let path = vault_meta_path(vault_path);
    std::fs::create_dir_all(path.parent().unwrap()).map_err(|e| e.to_string())?;
    std::fs::write(&path, B64.encode(salt)).map_err(|e| e.to_string())
}

fn load_vault_salt(vault_path: &Path) -> Result<Vec<u8>, String> {
    let path = vault_meta_path(vault_path);
    let b64 = std::fs::read_to_string(&path)
        .map_err(|_| "Vault not set up — call crypto_setup_vault first".to_string())?;
    B64.decode(b64.trim()).map_err(|e| e.to_string())
}

fn save_vault_verification(vault_path: &Path, blob: &EncryptedBlob) -> Result<(), String> {
    let path = vault_path.join(".vibo").join("vault.verify");
    let json = serde_json::to_string(blob).map_err(|e| e.to_string())?;
    std::fs::write(&path, json).map_err(|e| e.to_string())
}

fn load_vault_verification(vault_path: &Path) -> Result<EncryptedBlob, String> {
    let path = vault_path.join(".vibo").join("vault.verify");
    let json = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
    serde_json::from_str(&json).map_err(|e| e.to_string())
}

// ─────────────────────────────────────────
// REGISTER ALL COMMANDS in main.rs:
//
// .invoke_handler(tauri::generate_handler![
//     crypto::crypto_unlock,
//     crypto::crypto_lock,
//     crypto::crypto_is_unlocked,
//     crypto::crypto_setup_vault,
//     crypto::crypto_change_pin,
//     crypto::crypto_encrypt_note,
//     crypto::crypto_decrypt_note,
//     crypto::crypto_keystore_set,
//     crypto::crypto_keystore_get,
//     crypto::crypto_keystore_delete,
//     crypto::crypto_keystore_list,
// ])
//
// BIOMETRIC FLOW (frontend):
//   import { authenticate } from '@tauri-apps/plugin-biometric'
//   await authenticate('Unlock your vault')       // biometric gate
//   await invoke('crypto_unlock', { pin })        // then unlock with PIN
//   // vault is now unlocked, session key in memory
//   await invoke('crypto_decrypt_note', { blob }) // use freely
//   await invoke('crypto_lock')                   // lock on app background
// ─────────────────────────────────────────
