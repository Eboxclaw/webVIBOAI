// src-tauri/src/main.rs — ViBo
//
// Registers all Tauri commands and managed states.
// 115 commands across 9 modules.
//
// Module map:
//   notes.rs     22 commands — note CRUD, wikilinks, search, snapshots
//   kanban.rs    16 commands — boards, cards, subtasks, calendar sync
//   storage.rs   19 commands — SQLite index, embeddings, SRI cache, routing, memory, distillations
//   crypto.rs    11 commands — AES-256-GCM, Argon2, keystore, vault lock/unlock
//   graph.rs     12 commands — knowledge graph, paths, clusters, hubs
//   vault.rs      9 commands — encrypted notes (depends on crypto)
//   google.rs    14 commands — OAuth, Calendar CRUD, Gmail read-only
//   providers.rs  5 commands — LFM/Ollama/Anthropic/OpenRouter/Kimi/Minimax streaming
//   training.rs   7 commands — CCP/Exo compute scalers, Unsloth fine-tuning
//
// State map:
//   NotesState      { vault_path }
//   KanbanState     { vault_path }
//   StorageState    { db: Mutex<Connection>, vault_path }
//   CryptoState     { db: Mutex<Connection>, vault_path, session_key, vault_state }
//   GraphState      { vault_path, storage: Mutex<Connection> }
//   VaultState      { vault_path }
//   ProvidersState  { tor_enabled: Mutex<bool>, tor_proxy: String }
//   TrainingState   { vault_path, ccp_endpoint, exo_endpoint, jobs }
//
// Plugins:
//   tauri-plugin-biometric   → Android (fingerprint) + iOS (Face/Touch ID)
//   tauri-plugin-deep-link   → OAuth callback (vibo://oauth/callback)
//   tauri-plugin-fs          → filesystem access
//   tauri-plugin-shell       → spawn Python subprocesses (Unsloth)
//   tauri-plugin-notification → training done, overdue cards
//
// Cargo.toml dependencies needed:
//   tauri = { version = "2", features = ["protocol-asset"] }
//   tauri-plugin-biometric = "2"
//   tauri-plugin-deep-link = "2"
//   tauri-plugin-fs = "2"
//   tauri-plugin-shell = "2"
//   tauri-plugin-notification = "2"
//   rusqlite = { version = "0.31", features = ["bundled"] }
//   serde = { version = "1", features = ["derive"] }
//   serde_json = "1"
//   serde_yaml = "0.9"
//   aes-gcm = "0.10"
//   argon2 = "0.5"
//   rand = "0.8"
//   base64 = "0.22"
//   zeroize = "1"
//   reqwest = { version = "0.12", features = ["json", "stream", "socks"] }
//   tokio = { version = "1", features = ["full"] }
//   futures-util = "0.3"
//   chrono = { version = "0.4", features = ["serde"] }
//   uuid = { version = "1", features = ["v4"] }
//   regex = "1"
//   url = "2"
//   num_cpus = "1"

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod notes;
mod kanban;
mod storage;
mod crypto;
mod graph;
mod vault;
mod oauth;
mod google;
mod providers;
mod training;

use std::path::PathBuf;
use tauri::Manager;

// ─────────────────────────────────────────
// VAULT PATH RESOLUTION
// ─────────────────────────────────────────

fn resolve_vault_path(app: &tauri::App) -> PathBuf {
    // 1. Check for user-configured vault path in app data
    let config_path = app.path().app_data_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("vault_path.txt");

    if let Ok(path_str) = std::fs::read_to_string(&config_path) {
        let path = PathBuf::from(path_str.trim());
        if path.exists() {
            return path;
        }
    }

    // 2. Default: Documents/ViBo on all platforms
    let documents = app.path().document_dir()
        .unwrap_or_else(|_| PathBuf::from("."));
    let default_vault = documents.join("ViBo");
    std::fs::create_dir_all(&default_vault).ok();
    default_vault
}

// ─────────────────────────────────────────
// ENTRYPOINT
// ─────────────────────────────────────────

fn main() {
    tauri::Builder::default()

        // ── Plugins ──────────────────────────────────────────────────
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_deep_link::init())

        // Biometric: Android + iOS only
        // On desktop this plugin is a no-op — guard with #[cfg] if needed
        .plugin(tauri_plugin_biometric::Builder::new().build())

        // ── State setup ──────────────────────────────────────────────
        .setup(|app| {
            let vault_path = resolve_vault_path(app);

            // notes.rs
            app.manage(notes::NotesState {
                vault_path: vault_path.clone(),
            });

            // kanban.rs
            app.manage(kanban::KanbanState {
                vault_path: vault_path.clone(),
            });

            // storage.rs — SQLite with sqlite-vec
            let storage_state = storage::StorageState::new(&vault_path)
                .expect("Failed to init storage SQLite");
            app.manage(storage_state);

            // crypto.rs — keystore SQLite + session key
            let crypto_state = crypto::CryptoState::new(&vault_path)
                .expect("Failed to init crypto keystore");
            app.manage(crypto_state);

            // graph.rs — graph edges in SQLite
            let graph_state = graph::GraphState::new(&vault_path)
                .expect("Failed to init graph SQLite");
            app.manage(graph_state);

            // vault.rs — encrypted notes
            app.manage(vault::VaultState {
                vault_path: vault_path.clone(),
            });

            // providers.rs — inference + Tor
            app.manage(providers::ProvidersState::default());

            // training.rs — compute scalers + jobs
            app.manage(training::TrainingState::new(&vault_path));

            // oauth.rs — generic OAuth, pre-register Google
            let oauth_state = oauth::OAuthState::new();
            oauth_state.providers.lock().unwrap()
                .insert("google".to_string(), oauth::google_provider_config());
            app.manage(oauth_state);

            // Deep link handler — Google OAuth callback
            // Frontend receives "deep-link" event and calls google_auth_callback
            #[cfg(desktop)]
            {
                let handle = app.handle().clone();
                tauri_plugin_deep_link::register(
                    app.handle(),
                    move |request| {
                        handle.emit("deep-link", request.to_string()).ok();
                    },
                ).ok();
            }

            Ok(())
        })

        // ── Commands ─────────────────────────────────────────────────
        .invoke_handler(tauri::generate_handler![

            // notes.rs — 22 commands
            notes::note_create,
            notes::note_read,
            notes::note_write,
            notes::note_patch,
            notes::note_delete,
            notes::note_move,
            notes::note_rename,
            notes::note_list,
            notes::note_list_folder,
            notes::note_search,
            notes::note_search_tags,
            notes::note_get_frontmatter,
            notes::note_set_frontmatter,
            notes::note_get_links,
            notes::note_get_backlinks,
            notes::note_get_orphans,
            notes::note_get_graph,
            notes::note_daily_get,
            notes::note_snapshot,
            notes::note_restore,
            notes::note_list_snapshots,
            notes::note_stats,

            // kanban.rs — 16 commands
            kanban::kanban_create_board,
            kanban::kanban_get_board,
            kanban::kanban_list_boards,
            kanban::kanban_add_column,
            kanban::kanban_create_card,
            kanban::kanban_get_card,
            kanban::kanban_update_card,
            kanban::kanban_move_card,
            kanban::kanban_complete_subtask,
            kanban::kanban_archive_card,
            kanban::kanban_delete_card,
            kanban::kanban_create_from_calendar,
            kanban::kanban_get_due,
            kanban::kanban_get_overdue,
            kanban::kanban_get_by_event,
            kanban::kanban_search,

            // storage.rs — 19 commands
            storage::storage_index_note,
            storage::storage_remove_note,
            storage::storage_list_notes,
            storage::storage_get_unembedded,
            storage::storage_store_embedding,
            storage::storage_semantic_search,
            storage::storage_delete_embeddings,
            storage::storage_cache_lookup,
            storage::storage_cache_store,
            storage::storage_cache_clear,
            storage::storage_route_query,
            storage::storage_add_routing_signal,
            storage::storage_list_routing_signals,
            storage::storage_memory_set,
            storage::storage_memory_get,
            storage::storage_memory_get_session,
            storage::storage_memory_delete,
            storage::storage_store_distillation,
            storage::storage_get_distillations,

            // crypto.rs — 11 commands
            crypto::crypto_unlock,
            crypto::crypto_lock,
            crypto::crypto_is_unlocked,
            crypto::crypto_setup_vault,
            crypto::crypto_change_pin,
            crypto::crypto_encrypt_note,
            crypto::crypto_decrypt_note,
            crypto::crypto_keystore_set,
            crypto::crypto_keystore_get,
            crypto::crypto_keystore_delete,
            crypto::crypto_keystore_list,

            // graph.rs — 12 commands
            graph::graph_upsert_edge,
            graph::graph_remove_note,
            graph::graph_index_note,
            graph::graph_add_semantic_edges,
            graph::graph_add_tag_edges,
            graph::graph_get_full,
            graph::graph_get_local,
            graph::graph_find_path,
            graph::graph_get_orphans,
            graph::graph_get_hubs,
            graph::graph_get_stats,
            graph::graph_get_cluster,

            // vault.rs — 9 commands
            vault::vault_create,
            vault::vault_read,
            vault::vault_write,
            vault::vault_delete,
            vault::vault_list,
            vault::vault_search,
            vault::vault_snapshot,
            vault::vault_restore,
            vault::vault_count,

            // google.rs — 14 commands
            google::google_set_credentials,
            google::google_auth_start,
            google::google_auth_callback,
            google::google_auth_status,
            google::google_auth_revoke,
            google::google_calendar_list,
            google::google_calendar_events,
            google::google_calendar_today,
            google::google_calendar_create,
            google::google_calendar_update,
            google::google_calendar_delete,
            google::google_gmail_list,
            google::google_gmail_read,
            google::google_gmail_unread_count,

            // providers.rs — 5 commands
            providers::providers_list,
            providers::providers_tor_set,
            providers::providers_tor_status,
            providers::providers_stream,
            providers::providers_complete,

            // training.rs — 7 commands
            training::ccp_set_endpoint,
            training::exo_set_endpoint,
            training::compute_status,
            training::training_start,
            training::training_list_jobs,
            training::training_list_adapters,
            training::training_delete_adapter,
        ])
        .run(tauri::generate_context!())
        .expect("ViBo failed to start");
}
