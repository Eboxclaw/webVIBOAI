/// training.rs — ViBo Compute Scaling + Local Training
///
/// CCP  → CPU extension: offload inference/training to local desktop/server
/// Exo  → P2P compute: share inference/training across ViBo user devices
///
/// Both serve as compute scalers — phone stays cool, heavy work runs elsewhere.
/// Unsloth uses these as backends when available.
///
/// Events:
///   "training-progress"  { job_id, step, total_steps, loss }
///   "training-done"      { job_id, adapter_path }
///   "training-error"     { job_id, error }

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Mutex;
use tauri::{AppHandle, Emitter, State};
use tokio::io::AsyncBufReadExt;
use uuid::Uuid;

// ─────────────────────────────────────────
// TYPES
// ─────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ComputeStatus {
    pub backend: String,            // "local" | "ccp" | "exo"
    pub available: bool,
    pub endpoint: Option<String>,
    pub description: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Running,
    Done,
    Failed,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrainingJob {
    pub id: String,
    pub adapter_name: String,
    pub data_path: String,
    pub model_base: String,
    pub status: JobStatus,
    pub step: usize,
    pub total_steps: usize,
    pub loss: Option<f32>,
    pub adapter_path: Option<String>,
    pub error: Option<String>,
    pub backend: String,
    pub created_at: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainArgs {
    pub model_base: String,
    pub data_path: String,          // JSONL path in vault
    pub adapter_name: String,
    pub max_steps: Option<usize>,
    pub backend: Option<String>,    // "local" | "ccp" | "exo" — auto if None
}

pub struct TrainingState {
    pub vault_path: PathBuf,
    pub ccp_endpoint: Mutex<Option<String>>,    // e.g. "http://192.168.1.10:8080"
    pub exo_endpoint: Mutex<Option<String>>,    // e.g. "http://localhost:52415"
    pub jobs: Mutex<Vec<TrainingJob>>,
}

impl TrainingState {
    pub fn new(vault_path: &std::path::Path) -> Self {
        TrainingState {
            vault_path: vault_path.to_path_buf(),
            ccp_endpoint: Mutex::new(None),
            exo_endpoint: Mutex::new(None),
            jobs: Mutex::new(vec![]),
        }
    }
}

// ─────────────────────────────────────────
// BACKEND CONFIG
// ─────────────────────────────────────────

/// Set CCP endpoint — desktop/server running CCP daemon
#[tauri::command]
pub fn ccp_set_endpoint(
    state: State<TrainingState>,
    endpoint: Option<String>,
) -> Result<(), String> {
    *state.ccp_endpoint.lock().unwrap() = endpoint;
    Ok(())
}

/// Set Exo endpoint — P2P network
#[tauri::command]
pub fn exo_set_endpoint(
    state: State<TrainingState>,
    endpoint: Option<String>,
) -> Result<(), String> {
    *state.exo_endpoint.lock().unwrap() = endpoint;
    Ok(())
}

/// Check which compute backends are reachable
#[tauri::command]
pub async fn compute_status(state: State<'_, TrainingState>) -> Vec<ComputeStatus> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
        .unwrap();

    let mut backends = vec![
        ComputeStatus {
            backend: "local".to_string(),
            available: true,
            endpoint: None,
            description: "Device CPU — drains battery".to_string(),
        }
    ];

    if let Some(ep) = state.ccp_endpoint.lock().unwrap().clone() {
        let ok = client.get(format!("{}/health", ep)).send().await
            .map(|r| r.status().is_success()).unwrap_or(false);
        backends.push(ComputeStatus {
            backend: "ccp".to_string(),
            available: ok,
            endpoint: Some(ep),
            description: "Local desktop/server — fast, no battery drain".to_string(),
        });
    }

    if let Some(ep) = state.exo_endpoint.lock().unwrap().clone() {
        let ok = client.get(format!("{}/health", ep)).send().await
            .map(|r| r.status().is_success()).unwrap_or(false);
        backends.push(ComputeStatus {
            backend: "exo".to_string(),
            available: ok,
            endpoint: Some(ep),
            description: "Shared P2P compute with other ViBo users".to_string(),
        });
    }

    backends
}

fn best_backend(state: &TrainingState) -> (String, Option<String>) {
    if let Some(ep) = state.ccp_endpoint.lock().unwrap().clone() {
        return ("ccp".to_string(), Some(ep));
    }
    if let Some(ep) = state.exo_endpoint.lock().unwrap().clone() {
        return ("exo".to_string(), Some(ep));
    }
    ("local".to_string(), None)
}

// ─────────────────────────────────────────
// TRAINING
// ─────────────────────────────────────────

/// Start fine-tuning — auto picks best backend
#[tauri::command]
pub async fn training_start(
    state: State<'_, TrainingState>,
    app: AppHandle,
    args: TrainArgs,
) -> Result<TrainingJob, String> {
    let (backend, endpoint) = match args.backend.as_deref() {
        Some("ccp") => ("ccp".to_string(), state.ccp_endpoint.lock().unwrap().clone()),
        Some("exo") => ("exo".to_string(), state.exo_endpoint.lock().unwrap().clone()),
        _           => best_backend(&state),
    };

    let job_id = Uuid::new_v4().to_string();
    let output_dir = state.vault_path
        .join("models").join("adapters").join(&args.adapter_name)
        .to_string_lossy().to_string();

    let job = TrainingJob {
        id: job_id.clone(),
        adapter_name: args.adapter_name.clone(),
        data_path: args.data_path.clone(),
        model_base: args.model_base.clone(),
        status: JobStatus::Queued,
        step: 0,
        total_steps: args.max_steps.unwrap_or(100),
        loss: None,
        adapter_path: None,
        error: None,
        backend: backend.clone(),
        created_at: chrono::Utc::now().to_rfc3339(),
    };

    state.jobs.lock().unwrap().push(job.clone());

    match backend.as_str() {
        "ccp" | "exo" => {
            let ep = endpoint.ok_or("Backend endpoint not configured")?;
            spawn_remote_job(app, job_id, ep, args, output_dir);
        }
        _ => {
            spawn_local_job(app, job_id, args, output_dir, &state.vault_path)?;
        }
    }

    Ok(job)
}

fn spawn_remote_job(
    app: AppHandle,
    job_id: String,
    endpoint: String,
    args: TrainArgs,
    output_dir: String,
) {
    tokio::spawn(async move {
        let client = reqwest::Client::new();
        let resp = client.post(format!("{}/train", endpoint))
            .json(&serde_json::json!({
                "job_id": job_id,
                "model_base": args.model_base,
                "data_path": args.data_path,
                "adapter_name": args.adapter_name,
                "max_steps": args.max_steps.unwrap_or(100),
                "output_dir": output_dir,
            }))
            .send().await;

        match resp {
            Ok(r) if r.status().is_success() => {
                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                    let Ok(r) = client.get(format!("{}/train/{}", endpoint, job_id)).send().await
                        else { break };
                    let json: serde_json::Value = r.json().await.unwrap_or_default();
                    match json["status"].as_str() {
                        Some("done") => {
                            let _ = app.emit("training-done", serde_json::json!({
                                "job_id": job_id, "adapter_path": output_dir,
                            }));
                            break;
                        }
                        Some("failed") => {
                            let _ = app.emit("training-error", serde_json::json!({
                                "job_id": job_id, "error": json["error"],
                            }));
                            break;
                        }
                        _ => {
                            let _ = app.emit("training-progress", serde_json::json!({
                                "job_id": job_id,
                                "step": json["step"],
                                "total_steps": json["total_steps"],
                                "loss": json["loss"],
                            }));
                        }
                    }
                }
            }
            _ => {
                let _ = app.emit("training-error", serde_json::json!({
                    "job_id": job_id, "error": "Failed to reach compute backend",
                }));
            }
        }
    });
}

fn spawn_local_job(
    app: AppHandle,
    job_id: String,
    args: TrainArgs,
    output_dir: String,
    vault_path: &PathBuf,
) -> Result<(), String> {
    let script = format!(r#"
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{model}", max_seq_length=2048, load_in_4bit=True
)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16)
dataset = load_dataset("json", data_files="{data}", split="train")
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=dataset,
    dataset_text_field="text", max_seq_length=2048,
    args=TrainingArguments(
        max_steps={steps}, learning_rate=2e-4,
        output_dir="{out}", logging_steps=1
    ),
)
trainer.train()
model.save_pretrained("{out}")
"#,
        model = args.model_base,
        data  = args.data_path,
        steps = args.max_steps.unwrap_or(100),
        out   = output_dir,
    );

    let script_path = vault_path.join(".vibo").join("train_job.py");
    std::fs::create_dir_all(script_path.parent().unwrap()).map_err(|e| e.to_string())?;
    std::fs::write(&script_path, &script).map_err(|e| e.to_string())?;

    tokio::spawn(async move {
        let mut cmd = tokio::process::Command::new("python3");
        cmd.arg(&script_path).stdout(Stdio::piped());
        if let Ok(mut child) = cmd.spawn() {
            if let Some(stdout) = child.stdout.take() {
                let app2 = app.clone();
                let jid  = job_id.clone();
                tokio::spawn(async move {
                    let mut lines = tokio::io::BufReader::new(stdout).lines();
                    while let Ok(Some(line)) = lines.next_line().await {
                        if let Some((step, total, loss)) = parse_progress(&line) {
                            let _ = app2.emit("training-progress", serde_json::json!({
                                "job_id": jid, "step": step,
                                "total_steps": total, "loss": loss,
                            }));
                        }
                    }
                });
            }
            match child.wait().await {
                Ok(s) if s.success() => {
                    let _ = app.emit("training-done", serde_json::json!({
                        "job_id": job_id, "adapter_path": output_dir,
                    }));
                }
                _ => {
                    let _ = app.emit("training-error", serde_json::json!({
                        "job_id": job_id, "error": "Training failed",
                    }));
                }
            }
        }
    });

    Ok(())
}

fn parse_progress(line: &str) -> Option<(usize, usize, f32)> {
    let re = regex::Regex::new(
        r"[Ss]tep[:\s]+(\d+)[/\s]+(\d+).*loss[:\s=]+([\d.]+)"
    ).ok()?;
    let caps = re.captures(line)?;
    Some((caps[1].parse().ok()?, caps[2].parse().ok()?, caps[3].parse().ok()?))
}

// ─────────────────────────────────────────
// JOB + ADAPTER MANAGEMENT
// ─────────────────────────────────────────

#[tauri::command]
pub fn training_list_jobs(state: State<TrainingState>) -> Vec<TrainingJob> {
    state.jobs.lock().unwrap().clone()
}

#[tauri::command]
pub fn training_list_adapters(state: State<TrainingState>) -> Vec<String> {
    let dir = state.vault_path.join("models").join("adapters");
    if !dir.exists() { return vec![]; }
    std::fs::read_dir(&dir).ok().into_iter().flatten()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect()
}

#[tauri::command]
pub fn training_delete_adapter(
    state: State<TrainingState>,
    adapter_name: String,
) -> Result<(), String> {
    let path = state.vault_path.join("models").join("adapters").join(&adapter_name);
    std::fs::remove_dir_all(&path).map_err(|e| e.to_string())
}

// ─────────────────────────────────────────
// REGISTER in main.rs:
//
// .invoke_handler(tauri::generate_handler![
//     training::ccp_set_endpoint,
//     training::exo_set_endpoint,
//     training::compute_status,
//     training::training_start,
//     training::training_list_jobs,
//     training::training_list_adapters,
//     training::training_delete_adapter,
// ])
// ─────────────────────────────────────────
