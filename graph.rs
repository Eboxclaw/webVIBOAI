/// graph.rs — ViBo Knowledge Graph
///
/// Builds and queries the bidirectional knowledge graph from notes.
/// Nodes = notes. Edges = wikilinks + semantic similarity links.
///
/// Storage: graph edges live in SQLite (storage.rs).
/// Source of truth: .md files via notes.rs.
///
/// Graph is rebuilt incrementally — only changed notes are re-indexed.
///
/// Cargo.toml dependencies:
///   rusqlite = { version = "0.31", features = ["bundled"] }
///   serde = { version = "1", features = ["derive"] }
///   serde_json = "1"

use rusqlite::params;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::Mutex;
use tauri::State;

// ─────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GraphNode {
    pub id: String,                 // note relative path e.g. "folder/note.md"
    pub title: String,
    pub tags: Vec<String>,
    pub outbound_count: usize,      // links going out
    pub inbound_count: usize,       // backlinks coming in
    pub cluster: Option<String>,    // tag cluster if assigned
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GraphEdge {
    pub source: String,             // note id
    pub target: String,             // note id
    pub edge_type: EdgeType,
    pub resolved: bool,
    pub weight: f32,                // 1.0 for wikilink, 0.0-1.0 for semantic
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
pub enum EdgeType {
    Wikilink,   // explicit [[link]]
    Semantic,   // inferred by embedding similarity
    Tag,        // share the same tag
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GraphData {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub unresolved_links: usize,
    pub isolated_nodes: usize,      // nodes with no edges
    pub most_linked: Vec<(String, usize)>, // top 5 most linked notes
    pub clusters: Vec<ClusterInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ClusterInfo {
    pub tag: String,
    pub note_count: usize,
    pub internal_links: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PathResult {
    pub from: String,
    pub to: String,
    pub path: Vec<String>,          // note ids in order
    pub hops: usize,
    pub found: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LocalGraph {
    pub center: GraphNode,
    pub neighbours: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    pub depth: usize,
}

pub struct GraphState {
    pub vault_path: PathBuf,
    pub storage: Mutex<rusqlite::Connection>,
}

impl GraphState {
    pub fn new(vault_path: &std::path::Path) -> rusqlite::Result<Self> {
        let db_path = vault_path.join(".vibo").join("storage.db");
        let conn = rusqlite::Connection::open(&db_path)?;
        conn.execute_batch("
            CREATE TABLE IF NOT EXISTS graph_edges (
                source      TEXT NOT NULL,
                target      TEXT NOT NULL,
                edge_type   TEXT NOT NULL DEFAULT 'wikilink',
                resolved    INTEGER NOT NULL DEFAULT 1,
                weight      REAL NOT NULL DEFAULT 1.0,
                created_at  TEXT NOT NULL,
                PRIMARY KEY (source, target, edge_type)
            );
            CREATE INDEX IF NOT EXISTS idx_graph_source ON graph_edges(source);
            CREATE INDEX IF NOT EXISTS idx_graph_target ON graph_edges(target);
            CREATE INDEX IF NOT EXISTS idx_graph_type ON graph_edges(edge_type);
        ")?;
        Ok(GraphState {
            vault_path: vault_path.to_path_buf(),
            storage: Mutex::new(conn),
        })
    }
}

// ─────────────────────────────────────────
// INTERNAL HELPERS
// ─────────────────────────────────────────

fn now_iso() -> String {
    chrono::Utc::now().to_rfc3339()
}

fn edge_type_str(e: &EdgeType) -> &'static str {
    match e {
        EdgeType::Wikilink => "wikilink",
        EdgeType::Semantic => "semantic",
        EdgeType::Tag      => "tag",
    }
}

fn edge_type_from_str(s: &str) -> EdgeType {
    match s {
        "semantic" => EdgeType::Semantic,
        "tag"      => EdgeType::Tag,
        _          => EdgeType::Wikilink,
    }
}

// ─────────────────────────────────────────
// GRAPH BUILD COMMANDS
// ─────────────────────────────────────────

/// Upsert a single edge — called by notes.rs after note save
#[tauri::command]
pub fn graph_upsert_edge(
    state: State<GraphState>,
    source: String,
    target: String,
    edge_type: Option<String>,
    resolved: bool,
    weight: Option<f32>,
) -> Result<(), String> {
    let db = state.storage.lock().unwrap();
    let et = edge_type.unwrap_or_else(|| "wikilink".to_string());
    let w = weight.unwrap_or(1.0);
    db.execute(
        "INSERT OR REPLACE INTO graph_edges (source, target, edge_type, resolved, weight, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![source, target, et, resolved as i32, w, now_iso()],
    ).map_err(|e| e.to_string())?;
    Ok(())
}

/// Remove all edges from a note — called before re-indexing
#[tauri::command]
pub fn graph_remove_note(state: State<GraphState>, note_id: String) -> Result<(), String> {
    let db = state.storage.lock().unwrap();
    db.execute("DELETE FROM graph_edges WHERE source = ?1 OR target = ?1", params![note_id])
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Rebuild edges for a note from its wikilinks
/// Called automatically after note_write / note_create
#[tauri::command]
pub fn graph_index_note(
    state: State<GraphState>,
    note_id: String,
    wikilinks: Vec<crate::notes::WikiLink>,
) -> Result<(), String> {
    let db = state.storage.lock().unwrap();
    // Remove old edges for this note
    db.execute("DELETE FROM graph_edges WHERE source = ?1 AND edge_type = 'wikilink'",
        params![note_id]).map_err(|e| e.to_string())?;

    let now = now_iso();
    for link in &wikilinks {
        let target = link.resolved_path.clone().unwrap_or(link.target.clone());
        db.execute(
            "INSERT OR REPLACE INTO graph_edges (source, target, edge_type, resolved, weight, created_at)
             VALUES (?1, ?2, 'wikilink', ?3, 1.0, ?4)",
            params![note_id, target, link.resolved as i32, now],
        ).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Add semantic edges from embedding similarity results
/// Called by embedding pipeline after notes are indexed
#[tauri::command]
pub fn graph_add_semantic_edges(
    state: State<GraphState>,
    source: String,
    similar_notes: Vec<(String, f32)>,  // (note_id, similarity_score)
    threshold: Option<f32>,
) -> Result<(), String> {
    let min_score = threshold.unwrap_or(0.75);
    let db = state.storage.lock().unwrap();
    // Remove old semantic edges for this note
    db.execute("DELETE FROM graph_edges WHERE source = ?1 AND edge_type = 'semantic'",
        params![source]).map_err(|e| e.to_string())?;

    let now = now_iso();
    for (target, score) in similar_notes {
        if score < min_score || target == source { continue; }
        db.execute(
            "INSERT OR REPLACE INTO graph_edges (source, target, edge_type, resolved, weight, created_at)
             VALUES (?1, ?2, 'semantic', 1, ?3, ?4)",
            params![source, target, score, now],
        ).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Add tag-based edges between notes sharing the same tag
#[tauri::command]
pub fn graph_add_tag_edges(
    state: State<GraphState>,
    tag: String,
    note_ids: Vec<String>,
) -> Result<(), String> {
    if note_ids.len() < 2 { return Ok(()); }
    let db = state.storage.lock().unwrap();
    let now = now_iso();
    // Remove old tag edges for this tag group
    db.execute(
        "DELETE FROM graph_edges WHERE edge_type = 'tag' AND source IN (SELECT source FROM graph_edges WHERE edge_type = 'tag' AND target IN (SELECT target FROM graph_edges WHERE edge_type = 'tag'))",
        [],
    ).ok();

    for (i, source) in note_ids.iter().enumerate() {
        for target in note_ids.iter().skip(i + 1) {
            db.execute(
                "INSERT OR IGNORE INTO graph_edges (source, target, edge_type, resolved, weight, created_at)
                 VALUES (?1, ?2, 'tag', 1, 0.5, ?3)",
                params![source, target, now],
            ).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

// ─────────────────────────────────────────
// GRAPH QUERY COMMANDS
// ─────────────────────────────────────────

/// Full graph — all nodes and edges
/// Use for D3/Cytoscape rendering
#[tauri::command]
pub fn graph_get_full(
    state: State<GraphState>,
    edge_types: Option<Vec<String>>,    // filter by type, None = all
    resolved_only: Option<bool>,
) -> Result<GraphData, String> {
    let db = state.storage.lock().unwrap();
    let resolved = resolved_only.unwrap_or(false);

    let edges: Vec<GraphEdge> = {
        let mut stmt = db.prepare(
            "SELECT source, target, edge_type, resolved, weight FROM graph_edges ORDER BY edge_type, weight DESC"
        ).map_err(|e| e.to_string())?;
        stmt.query_map([], |row| {
            Ok(GraphEdge {
                source: row.get(0)?,
                target: row.get(1)?,
                edge_type: edge_type_from_str(&row.get::<_, String>(2)?),
                resolved: row.get::<_, i32>(3)? == 1,
                weight: row.get(4)?,
            })
        }).map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .filter(|e| {
            let type_ok = edge_types.as_ref()
                .map(|t| t.contains(&edge_type_str(&e.edge_type).to_string()))
                .unwrap_or(true);
            let resolved_ok = !resolved || e.resolved;
            type_ok && resolved_ok
        })
        .collect()
    };

    // Build node list from unique note ids in edges
    let mut node_ids: HashSet<String> = HashSet::new();
    let mut outbound: HashMap<String, usize> = HashMap::new();
    let mut inbound: HashMap<String, usize> = HashMap::new();

    for edge in &edges {
        node_ids.insert(edge.source.clone());
        node_ids.insert(edge.target.clone());
        *outbound.entry(edge.source.clone()).or_insert(0) += 1;
        *inbound.entry(edge.target.clone()).or_insert(0) += 1;
    }

    // Also get all indexed notes even if orphaned
    {
        let mut stmt = db.prepare("SELECT id, title, tags FROM notes_index")
            .map_err(|e| e.to_string())?;
        let rows: Vec<(String, String, String)> = stmt.query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        }).map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect();
        for (id, _, _) in &rows {
            node_ids.insert(id.clone());
        }
    }

    let nodes = {
        let mut stmt = db.prepare(
            "SELECT id, title, tags FROM notes_index WHERE id = ?1"
        ).map_err(|e| e.to_string())?;

        node_ids.iter().map(|id| {
            let (title, tags) = stmt.query_row(params![id], |row| {
                Ok((
                    row.get::<_, String>(1).unwrap_or_else(|_| id.clone()),
                    row.get::<_, String>(2).unwrap_or_else(|_| "[]".to_string()),
                ))
            }).unwrap_or_else(|_| (id.clone(), "[]".to_string()));

            let tags_vec: Vec<String> = serde_json::from_str(&tags).unwrap_or_default();

            GraphNode {
                id: id.clone(),
                title,
                tags: tags_vec,
                outbound_count: *outbound.get(id).unwrap_or(&0),
                inbound_count: *inbound.get(id).unwrap_or(&0),
                cluster: None,
            }
        }).collect()
    };

    Ok(GraphData { nodes, edges })
}

/// Local graph around a single note — depth levels of neighbours
#[tauri::command]
pub fn graph_get_local(
    state: State<GraphState>,
    note_id: String,
    depth: Option<usize>,
) -> Result<LocalGraph, String> {
    let max_depth = depth.unwrap_or(2);
    let db = state.storage.lock().unwrap();

    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<(String, usize)> = VecDeque::new();
    let mut edges: Vec<GraphEdge> = vec![];

    queue.push_back((note_id.clone(), 0));
    visited.insert(note_id.clone());

    while let Some((current, current_depth)) = queue.pop_front() {
        if current_depth >= max_depth { continue; }

        // Outbound
        let mut stmt = db.prepare(
            "SELECT source, target, edge_type, resolved, weight FROM graph_edges WHERE source = ?1"
        ).map_err(|e| e.to_string())?;
        let out_edges: Vec<GraphEdge> = stmt.query_map(params![current], |row| {
            Ok(GraphEdge {
                source: row.get(0)?,
                target: row.get(1)?,
                edge_type: edge_type_from_str(&row.get::<_, String>(2)?),
                resolved: row.get::<_, i32>(3)? == 1,
                weight: row.get(4)?,
            })
        }).map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect();

        for edge in out_edges {
            if !visited.contains(&edge.target) {
                visited.insert(edge.target.clone());
                queue.push_back((edge.target.clone(), current_depth + 1));
            }
            edges.push(edge);
        }

        // Inbound (backlinks)
        let mut stmt = db.prepare(
            "SELECT source, target, edge_type, resolved, weight FROM graph_edges WHERE target = ?1"
        ).map_err(|e| e.to_string())?;
        let in_edges: Vec<GraphEdge> = stmt.query_map(params![current], |row| {
            Ok(GraphEdge {
                source: row.get(0)?,
                target: row.get(1)?,
                edge_type: edge_type_from_str(&row.get::<_, String>(2)?),
                resolved: row.get::<_, i32>(3)? == 1,
                weight: row.get(4)?,
            })
        }).map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect();

        for edge in in_edges {
            if !visited.contains(&edge.source) {
                visited.insert(edge.source.clone());
                queue.push_back((edge.source.clone(), current_depth + 1));
            }
            edges.push(edge);
        }
    }

    // Build neighbour nodes
    let centre_node = get_node(&db, &note_id);
    let neighbours = visited.iter()
        .filter(|id| *id != &note_id)
        .map(|id| get_node(&db, id))
        .collect();

    Ok(LocalGraph {
        center: centre_node,
        neighbours,
        edges,
        depth: max_depth,
    })
}

/// Shortest path between two notes — BFS
#[tauri::command]
pub fn graph_find_path(
    state: State<GraphState>,
    from: String,
    to: String,
) -> Result<PathResult, String> {
    let db = state.storage.lock().unwrap();
    let mut visited: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<Vec<String>> = VecDeque::new();

    queue.push_back(vec![from.clone()]);
    visited.insert(from.clone());

    while let Some(path) = queue.pop_front() {
        let current = path.last().unwrap().clone();
        if current == to {
            let hops = path.len() - 1;
            return Ok(PathResult { from, to, path, hops, found: true });
        }
        if path.len() > 8 { continue; } // max depth guard

        let mut stmt = db.prepare(
            "SELECT target FROM graph_edges WHERE source = ?1 AND resolved = 1
             UNION
             SELECT source FROM graph_edges WHERE target = ?1 AND resolved = 1"
        ).map_err(|e| e.to_string())?;

        let neighbours: Vec<String> = stmt.query_map(params![current], |row| {
            row.get(0)
        }).map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect();

        for neighbour in neighbours {
            if !visited.contains(&neighbour) {
                visited.insert(neighbour.clone());
                let mut new_path = path.clone();
                new_path.push(neighbour);
                queue.push_back(new_path);
            }
        }
    }

    Ok(PathResult { from, to, path: vec![], hops: 0, found: false })
}

/// Notes not connected to any other note
#[tauri::command]
pub fn graph_get_orphans(state: State<GraphState>) -> Result<Vec<GraphNode>, String> {
    let db = state.storage.lock().unwrap();
    let mut stmt = db.prepare(
        "SELECT id, title, tags FROM notes_index
         WHERE id NOT IN (SELECT source FROM graph_edges)
         AND id NOT IN (SELECT target FROM graph_edges)"
    ).map_err(|e| e.to_string())?;

    let nodes = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
        ))
    }).map_err(|e| e.to_string())?
    .filter_map(|r| r.ok())
    .map(|(id, title, tags)| GraphNode {
        id,
        title,
        tags: serde_json::from_str(&tags).unwrap_or_default(),
        outbound_count: 0,
        inbound_count: 0,
        cluster: None,
    })
    .collect();

    Ok(nodes)
}

/// Most connected notes — sorted by total link count
#[tauri::command]
pub fn graph_get_hubs(
    state: State<GraphState>,
    limit: Option<usize>,
) -> Result<Vec<GraphNode>, String> {
    let db = state.storage.lock().unwrap();
    let lim = limit.unwrap_or(10) as i64;
    let mut stmt = db.prepare("
        SELECT n.id, n.title, n.tags,
               COUNT(DISTINCT e_out.target) as outbound,
               COUNT(DISTINCT e_in.source) as inbound
        FROM notes_index n
        LEFT JOIN graph_edges e_out ON e_out.source = n.id
        LEFT JOIN graph_edges e_in  ON e_in.target  = n.id
        GROUP BY n.id
        ORDER BY (outbound + inbound) DESC
        LIMIT ?1
    ").map_err(|e| e.to_string())?;

    let nodes = stmt.query_map(params![lim], |row| {
        Ok(GraphNode {
            id: row.get(0)?,
            title: row.get(1)?,
            tags: serde_json::from_str(&row.get::<_, String>(2).unwrap_or_default())
                .unwrap_or_default(),
            outbound_count: row.get::<_, usize>(3)?,
            inbound_count: row.get::<_, usize>(4)?,
            cluster: None,
        })
    }).map_err(|e| e.to_string())?
    .filter_map(|r| r.ok())
    .collect();

    Ok(nodes)
}

/// Graph stats for dashboard
#[tauri::command]
pub fn graph_get_stats(state: State<GraphState>) -> Result<GraphStats, String> {
    let db = state.storage.lock().unwrap();

    let node_count: usize = db.query_row(
        "SELECT COUNT(*) FROM notes_index", [], |r| r.get(0)
    ).unwrap_or(0);

    let edge_count: usize = db.query_row(
        "SELECT COUNT(*) FROM graph_edges", [], |r| r.get(0)
    ).unwrap_or(0);

    let unresolved_links: usize = db.query_row(
        "SELECT COUNT(*) FROM graph_edges WHERE resolved = 0", [], |r| r.get(0)
    ).unwrap_or(0);

    let isolated_nodes: usize = db.query_row(
        "SELECT COUNT(*) FROM notes_index WHERE id NOT IN (SELECT source FROM graph_edges) AND id NOT IN (SELECT target FROM graph_edges)",
        [], |r| r.get(0)
    ).unwrap_or(0);

    // Top 5 most linked
    let mut stmt = db.prepare("
        SELECT n.id, COUNT(*) as links
        FROM notes_index n
        JOIN graph_edges e ON e.source = n.id OR e.target = n.id
        GROUP BY n.id ORDER BY links DESC LIMIT 5
    ").map_err(|e| e.to_string())?;
    let most_linked: Vec<(String, usize)> = stmt.query_map([], |row| {
        Ok((row.get(0)?, row.get(1)?))
    }).map_err(|e| e.to_string())?
    .filter_map(|r| r.ok())
    .collect();

    // Tag clusters
    let mut tag_stmt = db.prepare("
        SELECT json_each.value as tag, COUNT(*) as cnt
        FROM notes_index, json_each(notes_index.tags)
        GROUP BY tag ORDER BY cnt DESC LIMIT 10
    ").map_err(|e| e.to_string())?;
    let clusters: Vec<ClusterInfo> = tag_stmt.query_map([], |row| {
        Ok(ClusterInfo {
            tag: row.get(0)?,
            note_count: row.get(1)?,
            internal_links: 0,
        })
    }).map_err(|e| e.to_string())?
    .filter_map(|r| r.ok())
    .collect();

    Ok(GraphStats {
        node_count,
        edge_count,
        unresolved_links,
        isolated_nodes,
        most_linked,
        clusters,
    })
}

/// Notes that share a specific tag
#[tauri::command]
pub fn graph_get_cluster(
    state: State<GraphState>,
    tag: String,
) -> Result<Vec<GraphNode>, String> {
    let db = state.storage.lock().unwrap();
    let mut stmt = db.prepare(
        "SELECT id, title, tags FROM notes_index WHERE json_search(tags, 'one', ?1) IS NOT NULL"
    ).map_err(|e| e.to_string())?;

    let nodes = stmt.query_map(params![tag], |row| {
        Ok(GraphNode {
            id: row.get(0)?,
            title: row.get(1)?,
            tags: serde_json::from_str(&row.get::<_, String>(2).unwrap_or_default())
                .unwrap_or_default(),
            outbound_count: 0,
            inbound_count: 0,
            cluster: Some(tag.clone()),
        })
    }).map_err(|e| e.to_string())?
    .filter_map(|r| r.ok())
    .collect();

    Ok(nodes)
}

// ─────────────────────────────────────────
// INTERNAL
// ─────────────────────────────────────────

fn get_node(db: &rusqlite::Connection, id: &str) -> GraphNode {
    db.query_row(
        "SELECT id, title, tags FROM notes_index WHERE id = ?1",
        params![id],
        |row| Ok(GraphNode {
            id: row.get(0)?,
            title: row.get(1)?,
            tags: serde_json::from_str(&row.get::<_, String>(2).unwrap_or_default())
                .unwrap_or_default(),
            outbound_count: 0,
            inbound_count: 0,
            cluster: None,
        }),
    ).unwrap_or_else(|_| GraphNode {
        id: id.to_string(),
        title: id.to_string(),
        tags: vec![],
        outbound_count: 0,
        inbound_count: 0,
        cluster: None,
    })
}

// ─────────────────────────────────────────
// REGISTER ALL COMMANDS in main.rs:
//
// .invoke_handler(tauri::generate_handler![
//     graph::graph_upsert_edge,
//     graph::graph_remove_note,
//     graph::graph_index_note,
//     graph::graph_add_semantic_edges,
//     graph::graph_add_tag_edges,
//     graph::graph_get_full,
//     graph::graph_get_local,
//     graph::graph_find_path,
//     graph::graph_get_orphans,
//     graph::graph_get_hubs,
//     graph::graph_get_stats,
//     graph::graph_get_cluster,
// ])
// ─────────────────────────────────────────
