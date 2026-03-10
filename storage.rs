/// storage.rs — ViBo SQLite Layer
///
/// Single .db file per vault. Source of truth stays in .md files.
/// SQLite is purely a fast query/index layer + SRI working memory.
///
/// Tables:
///   notes_index        → fast listing without reading .md files
///   embeddings         → 384-dim vectors (all-MiniLM-L6-v2) via sqlite-vec
///   semantic_cache     → cached query→result pairs by vector similarity
///   routing_signals    → keyword/regex patterns for SRI pre-LLM routing
///   agent_memory       → persistent context between agent sessions
///   distillations      → compact knowledge distilled from notes by agents
///
/// SRI query flow:
///   1. routing_signals  (~1ms)   keyword/regex fast match
///   2. semantic_cache   (~5ms)   cosine sim on cached queries
///   3. embeddings       (~20ms)  full semantic search on vault
///   → decision: local agent / parallel agents / cloud

use rusqlite::{Connection, Result as SqlResult, params};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tauri::State;
use chrono::{DateTime, Utc};

pub const EMBEDDING_DIM: usize = 384; // all-MiniLM-L6-v2

// ─────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NoteIndexRow {
    pub id: String,
    pub title: String,
    pub path: String,
    pub tags: String,           // JSON array string
    pub word_count: i64,
    pub modified_at: String,    // ISO 8601
    pub has_embedding: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SemanticSearchResult {
    pub note_id: String,
    pub title: String,
    pub chunk_text: String,
    pub score: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CacheHit {
    pub query: String,
    pub result_json: String,
    pub score: f32,
    pub hit_count: i64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RoutingSignal {
    pub id: i64,
    pub name: String,           // e.g. "calendar_intent"
    pub pattern: String,        // regex or keyword
    pub signal_type: String,    // "keyword" | "regex" | "domain"
    pub target_tool: String,    // e.g. "kanban_create_card"
    pub priority: i64,          // higher = checked first
    pub enabled: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RoutingMatch {
    pub signal_name: String,
    pub target_tool: String,
    pub confidence: f32,
    pub matched_pattern: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AgentMemory {
    pub key: String,
    pub value: String,          // JSON
    pub session: Option<String>,
    pub expires_at: Option<String>,
    pub created_at: String,
    pub modified_at: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Distillation {
    pub id: String,             // UUID
    pub distillation_type: String, // "tag_cluster" | "temporal" | "semantic_links" | "summary"
    pub source_ids: String,     // JSON array of note ids
    pub content_md: String,     // compact markdown output
    pub tags: String,           // JSON array
    pub created_at: String,
}

pub struct StorageState {
    pub db: Mutex<Connection>,
    pub vault_path: PathBuf,
}

// ─────────────────────────────────────────
// INIT
// ─────────────────────────────────────────

impl StorageState {
    pub fn new(vault_path: &Path) -> SqlResult<Self> {
        let db_path = vault_path.join(".vibo").join("storage.db");
        std::fs::create_dir_all(db_path.parent().unwrap())
            .map_err(|e| rusqlite::Error::InvalidPath(e.to_string().into()))?;

        let conn = Connection::open(&db_path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA foreign_keys=ON;")?;

        // Load sqlite-vec extension
        unsafe {
            conn.load_extension_enable()?;
            conn.load_extension(Path::new("sqlite-vec"), None)?;
            conn.load_extension_disable()?;
        }

        let state = StorageState {
            db: Mutex::new(conn),
            vault_path: vault_path.to_path_buf(),
        };
        state.migrate()?;
        state.seed_routing_signals()?;
        Ok(state)
    }

    fn migrate(&self) -> SqlResult<()> {
        let db = self.db.lock().unwrap();

        // notes_index
        db.execute_batch("
            CREATE TABLE IF NOT EXISTS notes_index (
                id          TEXT PRIMARY KEY,
                title       TEXT NOT NULL,
                path        TEXT NOT NULL,
                tags        TEXT NOT NULL DEFAULT '[]',
                word_count  INTEGER NOT NULL DEFAULT 0,
                modified_at TEXT NOT NULL,
                has_embedding INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_notes_modified ON notes_index(modified_at DESC);
            CREATE INDEX IF NOT EXISTS idx_notes_tags ON notes_index(tags);
        ")?;

        // embeddings via sqlite-vec virtual table
        db.execute_batch(&format!("
            CREATE VIRTUAL TABLE IF NOT EXISTS embeddings USING vec0(
                note_id     TEXT,
                chunk_index INTEGER,
                chunk_text  TEXT,
                embedding   float[{}]
            );
        ", EMBEDDING_DIM))?;

        // semantic_cache
        db.execute_batch(&format!("
            CREATE TABLE IF NOT EXISTS semantic_cache (
                id          TEXT PRIMARY KEY,
                query_text  TEXT NOT NULL,
                result_json TEXT NOT NULL,
                hit_count   INTEGER NOT NULL DEFAULT 1,
                created_at  TEXT NOT NULL,
                last_hit_at TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS cache_vectors USING vec0(
                cache_id    TEXT,
                embedding   float[{}]
            );
        ", EMBEDDING_DIM))?;

        // routing_signals
        db.execute_batch("
            CREATE TABLE IF NOT EXISTS routing_signals (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL UNIQUE,
                pattern     TEXT NOT NULL,
                signal_type TEXT NOT NULL DEFAULT 'keyword',
                target_tool TEXT NOT NULL,
                priority    INTEGER NOT NULL DEFAULT 0,
                enabled     INTEGER NOT NULL DEFAULT 1
            );
        ")?;

        // agent_memory
        db.execute_batch(&format!("
            CREATE TABLE IF NOT EXISTS agent_memory (
                key         TEXT NOT NULL,
                session     TEXT,
                value       TEXT NOT NULL,
                expires_at  TEXT,
                created_at  TEXT NOT NULL,
                modified_at TEXT NOT NULL,
                PRIMARY KEY (key, session)
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors USING vec0(
                memory_key  TEXT,
                embedding   float[{}]
            );
        ", EMBEDDING_DIM))?;

        // distillations
        db.execute_batch("
            CREATE TABLE IF NOT EXISTS distillations (
                id                  TEXT PRIMARY KEY,
                distillation_type   TEXT NOT NULL,
                source_ids          TEXT NOT NULL DEFAULT '[]',
                content_md          TEXT NOT NULL,
                tags                TEXT NOT NULL DEFAULT '[]',
                created_at          TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_distill_type ON distillations(distillation_type);
            CREATE INDEX IF NOT EXISTS idx_distill_tags ON distillations(tags);
        ")?;

        Ok(())
    }

    /// Seed default SRI routing signals for ViBo
    fn seed_routing_signals(&self) -> SqlResult<()> {
        let db = self.db.lock().unwrap();
        let signals: Vec<(&str, &str, &str, &str, i64)> = vec![
            // (name, pattern, type, target_tool, priority)
            ("create_note_intent",    r"(cria|escreve|nova nota|new note|add note)", "regex",   "note_create",              10),
            ("search_note_intent",    r"(encontra|search|procura|find|busca)",       "regex",   "note_search",              10),
            ("kanban_move_intent",    r"(move|muda|transfere|coloca em)",            "regex",   "kanban_move_card",         10),
            ("kanban_create_intent",  r"(task|tarefa|card|criar card|new task)",     "regex",   "kanban_create_card",       10),
            ("calendar_read_intent",  r"(calendário|calendar|evento|event|agenda)",  "regex",   "google_calendar_read",     9),
            ("calendar_write_intent", r"(marca|agendar|schedule|criar evento)",      "regex",   "google_calendar_write",    9),
            ("daily_note_intent",     r"(hoje|today|nota diária|daily note)",        "regex",   "note_daily_get",           8),
            ("encrypt_intent",        r"(encripta|encryp|lock|bloqueia|segredo)",    "regex",   "vault_encrypt",            10),
            ("overdue_intent",        r"(atrasado|overdue|em atraso|deadline)",      "keyword", "kanban_get_overdue",       7),
        ];

        for (name, pattern, sig_type, tool, priority) in signals {
            db.execute(
                "INSERT OR IGNORE INTO routing_signals (name, pattern, signal_type, target_tool, priority) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![name, pattern, sig_type, tool, priority],
            )?;
        }
        Ok(())
    }
}

// ─────────────────────────────────────────
// NOTES INDEX COMMANDS
// ─────────────────────────────────────────

/// Upsert a note into the index (called after note_write/note_create)
#[tauri::command]
pub fn storage_index_note(
    state: State<StorageState>,
    id: String,
    title: String,
    path: String,
    tags: Vec<String>,
    word_count: usize,
    modified_at: String,
) -> Result<(), String> {
    let db = state.db.lock().unwrap();
    let tags_json = serde_json::to_string(&tags).unwrap_or_default();
    db.execute(
        "INSERT OR REPLACE INTO notes_index (id, title, path, tags, word_count, modified_at, has_embedding)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6,
             COALESCE((SELECT has_embedding FROM notes_index WHERE id = ?1), 0))",
        params![id, title, path, tags_json, word_count as i64, modified_at],
    ).map_err(|e| e.to_string())?;
    Ok(())
}

/// Remove a note from the index
#[tauri::command]
pub fn storage_remove_note(state: State<StorageState>, id: String) -> Result<(), String> {
    let db = state.db.lock().unwrap();
    db.execute("DELETE FROM notes_index WHERE id = ?1", params![id])
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Fast list all indexed notes (no .md reads)
#[tauri::command]
pub fn storage_list_notes(state: State<StorageState>) -> Result<Vec<NoteIndexRow>, String> {
    let db = state.db.lock().unwrap();
    let mut stmt = db.prepare(
        "SELECT id, title, path, tags, word_count, modified_at, has_embedding FROM notes_index ORDER BY modified_at DESC"
    ).map_err(|e| e.to_string())?;
    let rows = stmt.query_map([], |row| {
        Ok(NoteIndexRow {
            id: row.get(0)?,
            title: row.get(1)?,
            path: row.get(2)?,
            tags: row.get(3)?,
            word_count: row.get(4)?,
            modified_at: row.get(5)?,
            has_embedding: row.get::<_, i32>(6)? == 1,
        })
    }).map_err(|e| e.to_string())?;
    rows.map(|r| r.map_err(|e| e.to_string())).collect()
}

/// Notes that don't have embeddings yet (for background embedding queue)
#[tauri::command]
pub fn storage_get_unembedded(state: State<StorageState>) -> Result<Vec<NoteIndexRow>, String> {
    let db = state.db.lock().unwrap();
    let mut stmt = db.prepare(
        "SELECT id, title, path, tags, word_count, modified_at, has_embedding FROM notes_index WHERE has_embedding = 0"
    ).map_err(|e| e.to_string())?;
    let rows = stmt.query_map([], |row| {
        Ok(NoteIndexRow {
            id: row.get(0)?,
            title: row.get(1)?,
            path: row.get(2)?,
            tags: row.get(3)?,
            word_count: row.get(4)?,
            modified_at: row.get(5)?,
            has_embedding: false,
        })
    }).map_err(|e| e.to_string())?;
    rows.map(|r| r.map_err(|e| e.to_string())).collect()
}

// ─────────────────────────────────────────
// EMBEDDINGS COMMANDS
// ─────────────────────────────────────────

/// Store embedding chunks for a note (called by embedding service after MiniLM inference)
#[tauri::command]
pub fn storage_store_embedding(
    state: State<StorageState>,
    note_id: String,
    chunk_index: usize,
    chunk_text: String,
    embedding: Vec<f32>,        // 384 floats from all-MiniLM-L6-v2
) -> Result<(), String> {
    if embedding.len() != EMBEDDING_DIM {
        return Err(format!("Expected {} dimensions, got {}", EMBEDDING_DIM, embedding.len()));
    }
    let db = state.db.lock().unwrap();
    let blob = floats_to_blob(&embedding);
    db.execute(
        "INSERT INTO embeddings (note_id, chunk_index, chunk_text, embedding) VALUES (?1, ?2, ?3, ?4)",
        params![note_id, chunk_index as i64, chunk_text, blob],
    ).map_err(|e| e.to_string())?;
    // Mark as embedded in index
    db.execute(
        "UPDATE notes_index SET has_embedding = 1 WHERE id = ?1",
        params![note_id],
    ).map_err(|e| e.to_string())?;
    Ok(())
}

/// Semantic search — returns top-k notes by cosine similarity
#[tauri::command]
pub fn storage_semantic_search(
    state: State<StorageState>,
    query_embedding: Vec<f32>,  // 384 floats
    top_k: Option<usize>,
) -> Result<Vec<SemanticSearchResult>, String> {
    if query_embedding.len() != EMBEDDING_DIM {
        return Err(format!("Expected {} dimensions, got {}", EMBEDDING_DIM, query_embedding.len()));
    }
    let k = top_k.unwrap_or(10);
    let db = state.db.lock().unwrap();
    let blob = floats_to_blob(&query_embedding);
    let mut stmt = db.prepare("
        SELECT e.note_id, n.title, e.chunk_text, e.distance
        FROM embeddings e
        JOIN notes_index n ON e.note_id = n.id
        WHERE embedding MATCH ?1 AND k = ?2
        ORDER BY e.distance ASC
    ").map_err(|e| e.to_string())?;

    let rows = stmt.query_map(params![blob, k as i64], |row| {
        Ok(SemanticSearchResult {
            note_id: row.get(0)?,
            title: row.get(1)?,
            chunk_text: row.get(2)?,
            score: 1.0 - row.get::<_, f32>(3)?, // convert distance to score
        })
    }).map_err(|e| e.to_string())?;
    rows.map(|r| r.map_err(|e| e.to_string())).collect()
}

/// Delete embeddings for a note (called before re-embedding after edit)
#[tauri::command]
pub fn storage_delete_embeddings(
    state: State<StorageState>,
    note_id: String,
) -> Result<(), String> {
    let db = state.db.lock().unwrap();
    db.execute("DELETE FROM embeddings WHERE note_id = ?1", params![note_id])
        .map_err(|e| e.to_string())?;
    db.execute("UPDATE notes_index SET has_embedding = 0 WHERE id = ?1", params![note_id])
        .map_err(|e| e.to_string())?;
    Ok(())
}

// ─────────────────────────────────────────
// SEMANTIC CACHE COMMANDS (SRI Layer 2)
// ─────────────────────────────────────────

/// Check if a similar query has been cached — returns hit if score >= threshold
#[tauri::command]
pub fn storage_cache_lookup(
    state: State<StorageState>,
    query_embedding: Vec<f32>,
    threshold: Option<f32>,     // default 0.92 — high similarity required
) -> Result<Option<CacheHit>, String> {
    let min_score = threshold.unwrap_or(0.92);
    let db = state.db.lock().unwrap();
    let blob = floats_to_blob(&query_embedding);

    let result = db.query_row("
        SELECT cv.cache_id, sc.query_text, sc.result_json, sc.hit_count, cv.distance
        FROM cache_vectors cv
        JOIN semantic_cache sc ON cv.cache_id = sc.id
        WHERE cv.embedding MATCH ?1 AND k = 1
        ORDER BY cv.distance ASC
        LIMIT 1
    ", params![blob], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, i64>(3)?,
            row.get::<_, f32>(4)?,
        ))
    });

    match result {
        Ok((cache_id, query_text, result_json, hit_count, distance)) => {
            let score = 1.0 - distance;
            if score >= min_score {
                // Increment hit count
                let now = Utc::now().to_rfc3339();
                let _ = db.execute(
                    "UPDATE semantic_cache SET hit_count = hit_count + 1, last_hit_at = ?1 WHERE id = ?2",
                    params![now, cache_id],
                );
                Ok(Some(CacheHit { query: query_text, result_json, score, hit_count: hit_count + 1 }))
            } else {
                Ok(None)
            }
        }
        Err(_) => Ok(None),
    }
}

/// Store a query result in cache
#[tauri::command]
pub fn storage_cache_store(
    state: State<StorageState>,
    query_text: String,
    query_embedding: Vec<f32>,
    result_json: String,
) -> Result<(), String> {
    let db = state.db.lock().unwrap();
    let id = uuid::Uuid::new_v4().to_string();
    let now = Utc::now().to_rfc3339();
    let blob = floats_to_blob(&query_embedding);

    db.execute(
        "INSERT INTO semantic_cache (id, query_text, result_json, hit_count, created_at, last_hit_at) VALUES (?1, ?2, ?3, 1, ?4, ?4)",
        params![id, query_text, result_json, now],
    ).map_err(|e| e.to_string())?;

    db.execute(
        "INSERT INTO cache_vectors (cache_id, embedding) VALUES (?1, ?2)",
        params![id, blob],
    ).map_err(|e| e.to_string())?;

    Ok(())
}

/// Clear stale cache entries (call periodically or when vault changes significantly)
#[tauri::command]
pub fn storage_cache_clear(state: State<StorageState>) -> Result<usize, String> {
    let db = state.db.lock().unwrap();
    let deleted = db.execute("DELETE FROM semantic_cache", []).map_err(|e| e.to_string())?;
    db.execute("DELETE FROM cache_vectors", []).map_err(|e| e.to_string())?;
    Ok(deleted)
}

// ─────────────────────────────────────────
// ROUTING SIGNALS COMMANDS (SRI Layer 1)
// ─────────────────────────────────────────

/// Fast keyword/regex match against routing signals — sub-1ms
/// Called first on every user message before any LLM call
#[tauri::command]
pub fn storage_route_query(
    state: State<StorageState>,
    query: String,
) -> Result<Vec<RoutingMatch>, String> {
    let db = state.db.lock().unwrap();
    let mut stmt = db.prepare(
        "SELECT name, pattern, signal_type, target_tool, priority FROM routing_signals WHERE enabled = 1 ORDER BY priority DESC"
    ).map_err(|e| e.to_string())?;

    let signals: Vec<RoutingSignal> = stmt.query_map([], |row| {
        Ok(RoutingSignal {
            id: 0,
            name: row.get(0)?,
            pattern: row.get(1)?,
            signal_type: row.get(2)?,
            target_tool: row.get(3)?,
            priority: row.get(4)?,
            enabled: true,
        })
    }).map_err(|e| e.to_string())?
    .filter_map(|r| r.ok())
    .collect();

    let query_lower = query.to_lowercase();
    let mut matches = vec![];

    for signal in signals {
        let matched = match signal.signal_type.as_str() {
            "keyword" => query_lower.contains(&signal.pattern.to_lowercase()),
            "regex" => {
                regex::Regex::new(&signal.pattern)
                    .map(|re| re.is_match(&query_lower))
                    .unwrap_or(false)
            }
            _ => false,
        };
        if matched {
            matches.push(RoutingMatch {
                signal_name: signal.name,
                target_tool: signal.target_tool,
                confidence: 1.0, // keyword/regex = certain match
                matched_pattern: signal.pattern,
            });
        }
    }
    Ok(matches)
}

/// Add a custom routing signal
#[tauri::command]
pub fn storage_add_routing_signal(
    state: State<StorageState>,
    name: String,
    pattern: String,
    signal_type: String,
    target_tool: String,
    priority: i64,
) -> Result<(), String> {
    let db = state.db.lock().unwrap();
    db.execute(
        "INSERT OR REPLACE INTO routing_signals (name, pattern, signal_type, target_tool, priority) VALUES (?1, ?2, ?3, ?4, ?5)",
        params![name, pattern, signal_type, target_tool, priority],
    ).map_err(|e| e.to_string())?;
    Ok(())
}

/// List all routing signals
#[tauri::command]
pub fn storage_list_routing_signals(state: State<StorageState>) -> Result<Vec<RoutingSignal>, String> {
    let db = state.db.lock().unwrap();
    let mut stmt = db.prepare(
        "SELECT id, name, pattern, signal_type, target_tool, priority, enabled FROM routing_signals ORDER BY priority DESC"
    ).map_err(|e| e.to_string())?;
    let rows = stmt.query_map([], |row| {
        Ok(RoutingSignal {
            id: row.get(0)?,
            name: row.get(1)?,
            pattern: row.get(2)?,
            signal_type: row.get(3)?,
            target_tool: row.get(4)?,
            priority: row.get(5)?,
            enabled: row.get::<_, i32>(6)? == 1,
        })
    }).map_err(|e| e.to_string())?;
    rows.map(|r| r.map_err(|e| e.to_string())).collect()
}

// ─────────────────────────────────────────
// AGENT MEMORY COMMANDS
// ─────────────────────────────────────────

/// Set a memory key (upsert)
#[tauri::command]
pub fn storage_memory_set(
    state: State<StorageState>,
    key: String,
    value: String,
    session: Option<String>,
    expires_at: Option<String>,
) -> Result<(), String> {
    let db = state.db.lock().unwrap();
    let now = Utc::now().to_rfc3339();
    let sess = session.clone().unwrap_or_default();
    db.execute(
        "INSERT OR REPLACE INTO agent_memory (key, session, value, expires_at, created_at, modified_at)
         VALUES (?1, ?2, ?3, ?4, COALESCE((SELECT created_at FROM agent_memory WHERE key = ?1 AND session = ?2), ?5), ?5)",
        params![key, sess, value, expires_at, now],
    ).map_err(|e| e.to_string())?;
    Ok(())
}

/// Get a memory key
#[tauri::command]
pub fn storage_memory_get(
    state: State<StorageState>,
    key: String,
    session: Option<String>,
) -> Result<Option<AgentMemory>, String> {
    let db = state.db.lock().unwrap();
    let sess = session.unwrap_or_default();
    let now = Utc::now().to_rfc3339();
    let result = db.query_row(
        "SELECT key, value, session, expires_at, created_at, modified_at FROM agent_memory
         WHERE key = ?1 AND session = ?2 AND (expires_at IS NULL OR expires_at > ?3)",
        params![key, sess, now],
        |row| Ok(AgentMemory {
            key: row.get(0)?,
            value: row.get(1)?,
            session: row.get(2)?,
            expires_at: row.get(3)?,
            created_at: row.get(4)?,
            modified_at: row.get(5)?,
        }),
    );
    match result {
        Ok(mem) => Ok(Some(mem)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.to_string()),
    }
}

/// Get all memory for a session
#[tauri::command]
pub fn storage_memory_get_session(
    state: State<StorageState>,
    session: String,
) -> Result<Vec<AgentMemory>, String> {
    let db = state.db.lock().unwrap();
    let now = Utc::now().to_rfc3339();
    let mut stmt = db.prepare(
        "SELECT key, value, session, expires_at, created_at, modified_at FROM agent_memory
         WHERE session = ?1 AND (expires_at IS NULL OR expires_at > ?2)
         ORDER BY modified_at DESC"
    ).map_err(|e| e.to_string())?;
    let rows = stmt.query_map(params![session, now], |row| {
        Ok(AgentMemory {
            key: row.get(0)?,
            value: row.get(1)?,
            session: row.get(2)?,
            expires_at: row.get(3)?,
            created_at: row.get(4)?,
            modified_at: row.get(5)?,
        })
    }).map_err(|e| e.to_string())?;
    rows.map(|r| r.map_err(|e| e.to_string())).collect()
}

/// Delete a memory key
#[tauri::command]
pub fn storage_memory_delete(
    state: State<StorageState>,
    key: String,
    session: Option<String>,
) -> Result<(), String> {
    let db = state.db.lock().unwrap();
    let sess = session.unwrap_or_default();
    db.execute("DELETE FROM agent_memory WHERE key = ?1 AND session = ?2", params![key, sess])
        .map_err(|e| e.to_string())?;
    Ok(())
}

// ─────────────────────────────────────────
// DISTILLATIONS COMMANDS
// ─────────────────────────────────────────

/// Store a distillation (compact knowledge from agents)
#[tauri::command]
pub fn storage_store_distillation(
    state: State<StorageState>,
    distillation_type: String,
    source_ids: Vec<String>,
    content_md: String,
    tags: Vec<String>,
) -> Result<Distillation, String> {
    let db = state.db.lock().unwrap();
    let id = uuid::Uuid::new_v4().to_string();
    let now = Utc::now().to_rfc3339();
    let source_json = serde_json::to_string(&source_ids).unwrap_or_default();
    let tags_json = serde_json::to_string(&tags).unwrap_or_default();

    db.execute(
        "INSERT INTO distillations (id, distillation_type, source_ids, content_md, tags, created_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![id, distillation_type, source_json, content_md, tags_json, now],
    ).map_err(|e| e.to_string())?;

    Ok(Distillation {
        id,
        distillation_type,
        source_ids: source_json,
        content_md,
        tags: tags_json,
        created_at: now,
    })
}

/// Get distillations by type
#[tauri::command]
pub fn storage_get_distillations(
    state: State<StorageState>,
    distillation_type: Option<String>,
    limit: Option<usize>,
) -> Result<Vec<Distillation>, String> {
    let db = state.db.lock().unwrap();
    let lim = limit.unwrap_or(20) as i64;
    let rows: Vec<Distillation> = if let Some(dtype) = distillation_type {
        let mut stmt = db.prepare(
            "SELECT id, distillation_type, source_ids, content_md, tags, created_at FROM distillations WHERE distillation_type = ?1 ORDER BY created_at DESC LIMIT ?2"
        ).map_err(|e| e.to_string())?;
        stmt.query_map(params![dtype, lim], |row| {
            Ok(Distillation {
                id: row.get(0)?,
                distillation_type: row.get(1)?,
                source_ids: row.get(2)?,
                content_md: row.get(3)?,
                tags: row.get(4)?,
                created_at: row.get(5)?,
            })
        }).map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect()
    } else {
        let mut stmt = db.prepare(
            "SELECT id, distillation_type, source_ids, content_md, tags, created_at FROM distillations ORDER BY created_at DESC LIMIT ?1"
        ).map_err(|e| e.to_string())?;
        stmt.query_map(params![lim], |row| {
            Ok(Distillation {
                id: row.get(0)?,
                distillation_type: row.get(1)?,
                source_ids: row.get(2)?,
                content_md: row.get(3)?,
                tags: row.get(4)?,
                created_at: row.get(5)?,
            })
        }).map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect()
    };
    Ok(rows)
}

// ─────────────────────────────────────────
// UTILS
// ─────────────────────────────────────────

fn floats_to_blob(floats: &[f32]) -> Vec<u8> {
    floats.iter().flat_map(|f| f.to_le_bytes()).collect()
}

// ─────────────────────────────────────────
// REGISTER ALL COMMANDS in main.rs:
//
// .invoke_handler(tauri::generate_handler![
//     storage::storage_index_note,
//     storage::storage_remove_note,
//     storage::storage_list_notes,
//     storage::storage_get_unembedded,
//     storage::storage_store_embedding,
//     storage::storage_semantic_search,
//     storage::storage_delete_embeddings,
//     storage::storage_cache_lookup,
//     storage::storage_cache_store,
//     storage::storage_cache_clear,
//     storage::storage_route_query,
//     storage::storage_add_routing_signal,
//     storage::storage_list_routing_signals,
//     storage::storage_memory_set,
//     storage::storage_memory_get,
//     storage::storage_memory_get_session,
//     storage::storage_memory_delete,
//     storage::storage_store_distillation,
//     storage::storage_get_distillations,
// ])
// ─────────────────────────────────────────
