use bytes::Bytes;
use futures::channel::mpsc as futures_mpsc;
use futures_util::{SinkExt, StreamExt};
use multer::Multipart;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use tokio::sync::mpsc;
use utoipa::{OpenApi, ToSchema};
use warp::{
    http::{header, StatusCode},
    ws::{Message, WebSocket},
    Filter, Rejection, Reply,
};

use crate::audio;
use crate::inference::{GenerationParams, TtsEngine};

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct SpeechRequest {
    /// Text to synthesize (`input` for OpenAI / LiveKit `sparktts.py` compatibility)
    #[serde(default)]
    pub input: Option<String>,
    /// Alias for `input` (`text` in `spark-tts-rs`)
    #[serde(default)]
    pub text: Option<String>,
    /// Voice ID (e.g. `"salt:lug_female_10"` or short ID `"lug_female_10"`)
    #[serde(default)]
    pub voice: Option<String>,
    /// Alias for `voice` (`speaker_id` in `spark-tts-rs`)
    #[serde(default)]
    pub speaker_id: Option<String>,
    /// Language hint (`"lug"`, `"swa"`, `"kin"`, `"ach"`, `"teo"`, `"eng"`)
    #[serde(default)]
    pub language: Option<String>,
    /// Segment identifier for WebSocket events (`segment_id` in `spark-tts-rs`)
    #[serde(default)]
    pub segment_id: Option<String>,
    /// Optional reference audio path for WebSocket voice cloning (`reference_audio_path` in `spark-tts-rs`)
    #[serde(default)]
    pub reference_audio_path: Option<String>,
    /// Optional reference transcript for voice cloning
    #[serde(default)]
    pub reference_text: Option<String>,
    /// Model name (accepted for OpenAI compatibility)
    #[serde(default)]
    pub model: Option<String>,
    /// Output format: `"wav"` (default) or `"pcm"`
    #[serde(default)]
    pub response_format: Option<String>,
    /// Sampling temperature (default: `0.7`)
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Nucleus sampling top-p (default: `0.9`, matching ArkttsModel)
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Top-k sampling (default: `50`, matching ArkttsModel)
    #[serde(default)]
    pub top_k: Option<usize>,
    /// Max semantic tokens per sentence chunk (default: `512`)
    #[serde(default)]
    pub max_new_tokens: Option<usize>,
    /// Random seed (default: `42`)
    #[serde(default)]
    pub seed: Option<u64>,
    /// Stream raw PCM chunks over HTTP when `true`
    #[serde(default)]
    pub stream: Option<bool>,
}

impl SpeechRequest {
    pub fn prompt_text(&self) -> Option<&str> {
        self.input
            .as_deref()
            .filter(|s| !s.trim().is_empty())
            .or_else(|| self.text.as_deref().filter(|s| !s.trim().is_empty()))
    }

    pub fn resolved_voice(&self) -> Option<&str> {
        self.voice
            .as_deref()
            .filter(|s| !s.trim().is_empty())
            .or_else(|| self.speaker_id.as_deref().filter(|s| !s.trim().is_empty()))
    }

    pub fn to_generation_params(&self) -> GenerationParams {
        let mut p = GenerationParams::default();
        if let Some(t) = self.temperature {
            p.temperature = t.clamp(0.05, 1.20);
        }
        if let Some(tp) = self.top_p {
            p.top_p = tp;
        }
        if let Some(tk) = self.top_k {
            p.top_k = tk;
        }
        if let Some(m) = self.max_new_tokens {
            p.max_new_tokens = m.clamp(16, 2048);
        }
        if let Some(s) = self.seed {
            p.seed = s;
        }
        p
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct WsEventMessage {
    #[serde(rename = "type")]
    pub msg_type: String,
    /// Included so both `spark-tts-rs` (`type == "end"`) and `sparktts.py` (`status == "done"`) work seamlessly
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segment_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_chunks: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<u32>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct HealthResponse {
    pub status: String,
    pub engine: String,
    pub provider: String,
    pub workers: usize,
    pub sample_rate: u32,
    pub native_sample_rate: u32,
    pub voices_loaded: usize,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct VoicesListResponse {
    pub voices: Vec<String>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct CloneVoiceResponse {
    pub status: String,
    pub voice: String,
    pub deduplicated: bool,
    pub message: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ErrorResponse {
    pub error: String,
}

#[derive(OpenApi)]
#[openapi(
    paths(
        health_doc,
        stream_speech_doc,
        clone_upload_doc,
        list_voices_doc,
        clone_voice_doc,
        delete_voice_doc
    ),
    components(schemas(
        SpeechRequest,
        HealthResponse,
        VoicesListResponse,
        CloneVoiceResponse,
        ErrorResponse
    )),
    tags(
        (name = "audio8-tts-rs", description = "Audio8-TTS Native Rust + LibTorch (tch-rs) API (compatible with spark-tts-rs)")
    )
)]
pub struct ApiDoc;

#[utoipa::path(
    get,
    path = "/health",
    responses((status = 200, description = "Server health and active execution provider", body = HealthResponse))
)]
#[allow(dead_code)]
fn health_doc() {}

#[utoipa::path(
    post,
    path = "/v1/audio/speech/stream",
    request_body = SpeechRequest,
    responses((status = 200, description = "Streamed 16-bit PCM audio chunks"))
)]
#[allow(dead_code)]
fn stream_speech_doc() {}

#[utoipa::path(
    post,
    path = "/v1/audio/speech/clone/upload",
    responses((status = 200, description = "Streamed 16-bit PCM audio cloned from uploaded reference audio"))
)]
#[allow(dead_code)]
fn clone_upload_doc() {}

#[utoipa::path(
    get,
    path = "/v1/voices",
    responses((status = 200, description = "List registered speaker IDs", body = VoicesListResponse))
)]
#[allow(dead_code)]
fn list_voices_doc() {}

#[utoipa::path(
    post,
    path = "/v1/voices/clone",
    responses((status = 200, description = "Clone and cache a speaker reference WAV", body = CloneVoiceResponse))
)]
#[allow(dead_code)]
fn clone_voice_doc() {}

#[utoipa::path(
    delete,
    path = "/v1/voices/{name}",
    params(("name" = String, Path, description = "Voice name to delete")),
    responses((status = 200, description = "Voice deleted"))
)]
#[allow(dead_code)]
fn delete_voice_doc() {}

pub fn build_routes(
    engine: Arc<TtsEngine>,
) -> impl Filter<Extract = (impl Reply,), Error = Rejection> + Clone {
    let cors = warp::cors()
        .allow_any_origin()
        .allow_methods(vec!["GET", "POST", "DELETE", "OPTIONS"])
        .allow_headers(vec!["Content-Type", "Authorization"]);

    let eng_health = engine.clone();
    let health_route = warp::path("health")
        .and(warp::get())
        .map(move || {
            let voices = eng_health.list_voices();
            let resp = HealthResponse {
                status: "ok".to_string(),
                engine: "audio8-tts-rs".to_string(),
                provider: eng_health.active_provider.clone(),
                workers: eng_health.config.num_workers,
                sample_rate: eng_health.output_sample_rate(),
                native_sample_rate: eng_health.native_sample_rate(),
                voices_loaded: voices.len(),
            };
            warp::reply::json(&resp)
        });

    let openapi_route = warp::path!("api-docs" / "openapi.json")
        .and(warp::get())
        .map(|| warp::reply::json(&ApiDoc::openapi()));

    let openapi_root_route = warp::path("openapi.json")
        .and(warp::path::end())
        .and(warp::get())
        .map(|| warp::reply::json(&ApiDoc::openapi()));

    let docs_route = warp::path("docs").and(warp::get()).map(|| {
        let html = r#"<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Audio8-TTS-RS API Docs</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css" />
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    window.onload = () => {
      window.ui = SwaggerUIBundle({ url: '/openapi.json', dom_id: '#swagger-ui' });
    };
  </script>
</body>
</html>"#;
        warp::reply::html(html)
    });

    // GET /v1/voices
    let eng_voices = engine.clone();
    let list_voices_route = warp::path!("v1" / "voices")
        .and(warp::get())
        .map(move || {
            let voices = eng_voices.list_voices();
            warp::reply::json(&VoicesListResponse { voices })
        });

    // DELETE /v1/voices/{name}
    let eng_del = engine.clone();
    let delete_voice_route = warp::path!("v1" / "voices" / String)
        .and(warp::delete())
        .map(move |name: String| {
            let decoded = urlencoding_decode(&name);
            match eng_del.delete_voice(&decoded) {
                Ok(()) => warp::reply::with_status(
                    warp::reply::json(&serde_json::json!({
                        "status": "deleted",
                        "voice": decoded
                    })),
                    StatusCode::OK,
                ),
                Err(e) => warp::reply::with_status(
                    warp::reply::json(&ErrorResponse {
                        error: e.to_string(),
                    }),
                    StatusCode::NOT_FOUND,
                ),
            }
        });

    // POST /v1/voices/clone (register & persist a cloned voice)
    let eng_clone = engine.clone();
    let clone_voice_route = warp::path!("v1" / "voices" / "clone")
        .and(warp::post())
        .and(warp::header::<String>("content-type"))
        .and(warp::body::bytes())
        .and_then(move |content_type: String, body: Bytes| {
            let eng = eng_clone.clone();
            async move { handle_clone_voice(eng, content_type, body).await }
        });

    // POST /v1/audio/speech/clone/upload (ad-hoc voice cloning + streaming PCM, matching `spark-tts-rs`)
    let eng_clone_upload = engine.clone();
    let clone_upload_route = warp::path!("v1" / "audio" / "speech" / "clone" / "upload")
        .and(warp::post())
        .and(warp::header::<String>("content-type"))
        .and(warp::body::bytes())
        .and_then(move |content_type: String, body: Bytes| {
            let eng = eng_clone_upload.clone();
            async move { handle_clone_upload_stream(eng, content_type, body).await }
        });

    // POST /v1/audio/speech (and `/tts` alias)
    let eng_speech = engine.clone();
    let speech_v1 = warp::path!("v1" / "audio" / "speech")
        .and(warp::post())
        .and(warp::body::json())
        .and_then(move |req: SpeechRequest| {
            let eng = eng_speech.clone();
            async move { handle_speech_http(eng, req).await }
        });

    let eng_tts = engine.clone();
    let tts_alias = warp::path("tts")
        .and(warp::path::end())
        .and(warp::post())
        .and(warp::body::json())
        .and_then(move |req: SpeechRequest| {
            let eng = eng_tts.clone();
            async move { handle_speech_http(eng, req).await }
        });

    // POST /v1/audio/speech/stream
    let eng_stream = engine.clone();
    let speech_stream_route = warp::path!("v1" / "audio" / "speech" / "stream")
        .and(warp::post())
        .and(warp::body::json())
        .and_then(move |req: SpeechRequest| {
            let eng = eng_stream.clone();
            async move { handle_speech_stream_http(eng, req).await }
        });

    // WebSocket routes:
    // - /v1/audio/speech/stream/ws (primary WS streaming route matching spark-tts-rs)
    // - /v1/audio/speech/clone/ws  (WS voice cloning route matching spark-tts-rs)
    // - /v1/audio/speech/ws
    let eng_ws_stream = engine.clone();
    let ws_stream_route = warp::path!("v1" / "audio" / "speech" / "stream" / "ws")
        .and(warp::ws())
        .map(move |ws: warp::ws::Ws| {
            let eng = eng_ws_stream.clone();
            ws.on_upgrade(move |socket| handle_ws_client(socket, eng))
        });

    let eng_ws_clone = engine.clone();
    let ws_clone_route = warp::path!("v1" / "audio" / "speech" / "clone" / "ws")
        .and(warp::ws())
        .map(move |ws: warp::ws::Ws| {
            let eng = eng_ws_clone.clone();
            ws.on_upgrade(move |socket| handle_ws_client(socket, eng))
        });

    let eng_ws2 = engine;
    let ws_v1_route = warp::path!("v1" / "audio" / "speech" / "ws")
        .and(warp::ws())
        .map(move |ws: warp::ws::Ws| {
            let eng = eng_ws2.clone();
            ws.on_upgrade(move |socket| handle_ws_client(socket, eng))
        });

    health_route
        .or(openapi_route)
        .or(openapi_root_route)
        .or(docs_route)
        .or(list_voices_route)
        .or(clone_voice_route)
        .or(delete_voice_route)
        .or(clone_upload_route)
        .or(ws_stream_route)
        .or(ws_clone_route)
        .or(ws_v1_route)
        .or(speech_stream_route)
        .or(speech_v1)
        .or(tts_alias)
        .with(cors)
}

pub async fn run_server(engine: Arc<TtsEngine>, addr: std::net::SocketAddr) {
    println!("============================================================");
    println!("Audio8-TTS-RS server listening on http://{}", addr);
    println!("  Swagger UI Docs : http://{}/docs", addr);
    println!("  Health Check    : http://{}/health", addr);
    println!("  HTTP Speech     : POST http://{}/v1/audio/speech", addr);
    println!("  HTTP Stream     : POST http://{}/v1/audio/speech/stream", addr);
    println!("  WebSocket Stream: ws://{}/v1/audio/speech/stream/ws", addr);
    println!("============================================================");
    let routes = build_routes(engine);
    warp::serve(routes).run(addr).await;
}

async fn handle_speech_http(
    engine: Arc<TtsEngine>,
    req: SpeechRequest,
) -> Result<Box<dyn Reply>, Infallible> {
    // Pure streaming mode matching spark-tts-rs (never buffers in memory to prevent timeouts)
    handle_speech_stream_http(engine, req).await
}

async fn handle_speech_stream_http(
    engine: Arc<TtsEngine>,
    req: SpeechRequest,
) -> Result<Box<dyn Reply>, Infallible> {
    let Some(prompt_text) = req.prompt_text().map(|s| s.to_string()) else {
        let err = warp::reply::with_status(
            warp::reply::json(&ErrorResponse {
                error: "Missing 'input' or 'text' field in request body".to_string(),
            }),
            StatusCode::BAD_REQUEST,
        );
        return Ok(Box::new(err));
    };

    let voice = req.resolved_voice().map(|s| s.to_string());
    let params = req.to_generation_params();
    let out_sr = engine.output_sample_rate();

    let (tx, rx) = futures_mpsc::unbounded::<Result<Bytes, Infallible>>();
    let abort_flag = Arc::new(AtomicBool::new(false));
    let abort_for_worker = abort_flag;

    tokio::task::spawn_blocking(move || {
        let res = engine.synthesize_streaming(
            &prompt_text,
            voice.as_deref(),
            params,
            Some(abort_for_worker.clone()),
            |chunk| {
                let pcm = audio::samples_to_pcm16le_bytes(&chunk);
                if tx.unbounded_send(Ok(Bytes::from(pcm))).is_err() {
                    abort_for_worker.store(true, Ordering::Relaxed);
                    return Ok(false);
                }
                Ok(true)
            },
        );
        if let Err(e) = res {
            eprintln!("Streaming synthesis error: {}", e);
        }
    });

    let body = warp::hyper::Body::wrap_stream(rx);
    let resp = warp::http::Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "audio/pcm")
        .header("X-Sample-Rate", out_sr.to_string())
        .header("X-Bit-Depth", "16")
        .header("X-Channels", "1")
        .body(body)
        .unwrap();

    Ok(Box::new(resp))
}

async fn handle_clone_upload_stream(
    engine: Arc<TtsEngine>,
    content_type: String,
    body: Bytes,
) -> Result<Box<dyn Reply>, Infallible> {
    let boundary = match multer::parse_boundary(&content_type) {
        Ok(b) => b,
        Err(e) => {
            let err = warp::reply::with_status(
                warp::reply::json(&ErrorResponse {
                    error: format!("Invalid multipart content-type: {}", e),
                }),
                StatusCode::BAD_REQUEST,
            );
            return Ok(Box::new(err));
        }
    };

    let stream = futures_util::stream::once(async move { Ok::<Bytes, Infallible>(body) });
    let mut multipart = Multipart::new(stream, boundary);

    let mut text: Option<String> = None;
    let mut reference_text: Option<String> = None;
    let mut language: Option<String> = None;
    let mut temperature: Option<f32> = None;
    let mut wav_bytes: Option<Vec<u8>> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let field_name = field.name().unwrap_or("").to_string();
        match field_name.as_str() {
            "text" | "input" => {
                if let Ok(v) = field.text().await {
                    text = Some(v);
                }
            }
            "reference_text" | "transcript" => {
                if let Ok(v) = field.text().await {
                    reference_text = Some(v);
                }
            }
            "language" | "lang" => {
                if let Ok(v) = field.text().await {
                    language = Some(v);
                }
            }
            "temperature" => {
                if let Ok(v) = field.text().await {
                    if let Ok(t) = v.trim().parse::<f32>() {
                        temperature = Some(t);
                    }
                }
            }
            "reference_audio" | "file" | "audio" => {
                if let Ok(b) = field.bytes().await {
                    wav_bytes = Some(b.to_vec());
                }
            }
            _ => {}
        }
    }

    let Some(prompt_text) = text.filter(|s| !s.trim().is_empty()) else {
        let err = warp::reply::with_status(
            warp::reply::json(&ErrorResponse {
                error: "Missing 'text' parameter in multipart upload".to_string(),
            }),
            StatusCode::BAD_REQUEST,
        );
        return Ok(Box::new(err));
    };

    let Some(audio_bytes) = wav_bytes.filter(|b| !b.is_empty()) else {
        let err = warp::reply::with_status(
            warp::reply::json(&ErrorResponse {
                error: "Missing 'reference_audio' file in multipart upload".to_string(),
            }),
            StatusCode::BAD_REQUEST,
        );
        return Ok(Box::new(err));
    };

    let mut params = GenerationParams::default();
    if let Some(t) = temperature {
        params.temperature = t.clamp(0.05, 0.90);
    }

    let out_sr = engine.output_sample_rate();
    let (tx, rx) = futures_mpsc::unbounded::<Result<Bytes, Infallible>>();
    let abort_flag = Arc::new(AtomicBool::new(false));
    let abort_for_worker = abort_flag;

    tokio::task::spawn_blocking(move || {
        let res = engine.synthesize_cloned_bytes_streaming(
            &prompt_text,
            &audio_bytes,
            reference_text.as_deref(),
            language.as_deref(),
            params,
            Some(abort_for_worker.clone()),
            |chunk| {
                let pcm = audio::samples_to_pcm16le_bytes(&chunk);
                if tx.unbounded_send(Ok(Bytes::from(pcm))).is_err() {
                    abort_for_worker.store(true, Ordering::Relaxed);
                    return Ok(false);
                }
                Ok(true)
            },
        );
        if let Err(e) = res {
            eprintln!("Clone upload streaming error: {}", e);
        }
    });

    let body = warp::hyper::Body::wrap_stream(rx);
    let resp = warp::http::Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "audio/pcm")
        .header("X-Sample-Rate", out_sr.to_string())
        .header("X-Bit-Depth", "16")
        .header("X-Channels", "1")
        .body(body)
        .unwrap();

    Ok(Box::new(resp))
}

async fn handle_ws_client(ws: WebSocket, engine: Arc<TtsEngine>) {
    let (mut ws_sender, mut ws_receiver) = ws.split();
    let (tx_ws_out, mut rx_ws_out) = mpsc::unbounded_channel::<Message>();

    let writer_handle = tokio::spawn(async move {
        while let Some(msg) = rx_ws_out.recv().await {
            if ws_sender.send(msg).await.is_err() {
                break;
            }
        }
    });

    let request_queue = Arc::new(std::sync::Mutex::new(
        std::collections::VecDeque::<(SpeechRequest, Arc<AtomicBool>)>::new(),
    ));
    let notify = Arc::new(tokio::sync::Notify::new());
    let current_active_abort_flag = Arc::new(std::sync::Mutex::new(None::<Arc<AtomicBool>>));

    let request_queue_clone = request_queue.clone();
    let notify_clone = notify.clone();
    let active_abort_clone = current_active_abort_flag.clone();
    let engine_clone = engine.clone();
    let tx_ws_out_clone = tx_ws_out.clone();

    let worker_handle = tokio::spawn(async move {
        loop {
            let next = {
                let mut queue = request_queue_clone.lock().unwrap();
                queue.pop_front()
            };

            if let Some((req, abort_flag)) = next {
                {
                    let mut active = active_abort_clone.lock().unwrap();
                    *active = Some(abort_flag.clone());
                }

                run_ws_request(req, engine_clone.clone(), tx_ws_out_clone.clone(), abort_flag).await;

                {
                    let mut active = active_abort_clone.lock().unwrap();
                    *active = None;
                }
            } else {
                notify_clone.notified().await;
            }
        }
    });

    while let Some(msg_res) = ws_receiver.next().await {
        let Ok(msg) = msg_res else {
            break;
        };
        if msg.is_close() {
            break;
        }
        if !msg.is_text() {
            continue;
        }
        let Ok(text_payload) = msg.to_str() else {
            continue;
        };
        let trimmed = text_payload.trim();
        if trimmed == "abort" || trimmed == "interrupt" {
            {
                let mut queue = request_queue.lock().unwrap();
                queue.clear();
            }
            {
                let mut active = current_active_abort_flag.lock().unwrap();
                if let Some(flag) = active.take() {
                    flag.store(true, Ordering::Relaxed);
                }
            }
            continue;
        }

        match serde_json::from_str::<SpeechRequest>(trimmed) {
            Ok(req) => {
                let abort_flag = Arc::new(AtomicBool::new(false));
                {
                    let mut queue = request_queue.lock().unwrap();
                    queue.push_back((req, abort_flag));
                }
                notify.notify_one();
            }
            Err(e) => {
                let err_msg = serde_json::json!({
                    "type": "error",
                    "error": format!("Invalid JSON: {}", e)
                });
                let _ = tx_ws_out.send(Message::text(err_msg.to_string()));
            }
        }
    }

    {
        let mut queue = request_queue.lock().unwrap();
        queue.clear();
    }
    {
        let mut active = current_active_abort_flag.lock().unwrap();
        if let Some(flag) = active.take() {
            flag.store(true, Ordering::Relaxed);
        }
    }

    writer_handle.abort();
    worker_handle.abort();
}

async fn run_ws_request(
    req: SpeechRequest,
    engine: Arc<TtsEngine>,
    tx_ws_out: mpsc::UnboundedSender<Message>,
    abort_flag: Arc<AtomicBool>,
) {
    let segment_id = req
        .segment_id
        .clone()
        .unwrap_or_else(|| "default".to_string());

    let Some(prompt_text) = req.prompt_text().map(|s| s.to_string()) else {
        let err = WsEventMessage {
            msg_type: "error".to_string(),
            segment_id: Some(segment_id),
            message: Some("Missing 'input' or 'text' field".to_string()),
            ..Default::default()
        };
        if let Ok(s) = serde_json::to_string(&err) {
            let _ = tx_ws_out.send(Message::text(s));
        }
        return;
    };

    let sample_rate = engine.output_sample_rate();
    let start_msg = WsEventMessage {
        msg_type: "start".to_string(),
        segment_id: Some(segment_id.clone()),
        sample_rate: Some(sample_rate),
        ..Default::default()
    };
    if let Ok(s) = serde_json::to_string(&start_msg) {
        if tx_ws_out.send(Message::text(s)).is_err() {
            return;
        }
    }

    let voice = req.resolved_voice().map(|s| s.to_string());
    let ref_audio_path = req.reference_audio_path.clone();
    let ref_text = req.reference_text.clone();
    let lang = req.language.clone();
    let params = req.to_generation_params();
    let abort_worker = abort_flag.clone();
    let tx_chunks = tx_ws_out.clone();

    let res = tokio::task::spawn_blocking(move || {
        if let Some(ref path_str) = ref_audio_path {
            let path = PathBuf::from(path_str);
            engine.synthesize_cloned_path_streaming(
                &prompt_text,
                &path,
                ref_text.as_deref(),
                lang.as_deref(),
                params,
                Some(abort_worker.clone()),
                |chunk| {
                    let pcm = audio::samples_to_pcm16le_bytes(&chunk);
                    if tx_chunks.send(Message::binary(pcm)).is_err() {
                        abort_worker.store(true, Ordering::Relaxed);
                        return Ok(false);
                    }
                    Ok(true)
                },
            )
        } else {
            engine.synthesize_streaming(
                &prompt_text,
                voice.as_deref(),
                params,
                Some(abort_worker.clone()),
                |chunk| {
                    let pcm = audio::samples_to_pcm16le_bytes(&chunk);
                    if tx_chunks.send(Message::binary(pcm)).is_err() {
                        abort_worker.store(true, Ordering::Relaxed);
                        return Ok(false);
                    }
                    Ok(true)
                },
            )
        }
    })
    .await;

    match res {
        Ok(Ok(())) => {
            let end_msg = WsEventMessage {
                msg_type: "end".to_string(),
                status: Some("done".to_string()),
                segment_id: Some(segment_id),
                ..Default::default()
            };
            if let Ok(s) = serde_json::to_string(&end_msg) {
                let _ = tx_ws_out.send(Message::text(s));
            }
        }
        Ok(Err(e)) => {
            let err_msg = WsEventMessage {
                msg_type: "error".to_string(),
                segment_id: Some(segment_id),
                message: Some(e.to_string()),
                ..Default::default()
            };
            if let Ok(s) = serde_json::to_string(&err_msg) {
                let _ = tx_ws_out.send(Message::text(s));
            }
        }
        Err(e) => {
            let err_msg = WsEventMessage {
                msg_type: "error".to_string(),
                segment_id: Some(segment_id),
                message: Some(format!("Worker panic: {}", e)),
                ..Default::default()
            };
            if let Ok(s) = serde_json::to_string(&err_msg) {
                let _ = tx_ws_out.send(Message::text(s));
            }
        }
    }
}

async fn handle_clone_voice(
    engine: Arc<TtsEngine>,
    content_type: String,
    body: Bytes,
) -> Result<Box<dyn Reply>, Infallible> {
    let boundary = match multer::parse_boundary(&content_type) {
        Ok(b) => b,
        Err(e) => {
            let err = warp::reply::with_status(
                warp::reply::json(&ErrorResponse {
                    error: format!("Invalid multipart content-type: {}", e),
                }),
                StatusCode::BAD_REQUEST,
            );
            return Ok(Box::new(err));
        }
    };

    let stream = futures_util::stream::once(async move { Ok::<Bytes, Infallible>(body) });
    let mut multipart = Multipart::new(stream, boundary);

    let mut name: Option<String> = None;
    let mut reference_text = String::new();
    let mut language = "eng".to_string();
    let mut wav_bytes: Option<Vec<u8>> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let field_name = field.name().unwrap_or("").to_string();
        match field_name.as_str() {
            "name" | "voice" => {
                if let Ok(txt) = field.text().await {
                    name = Some(txt.trim().to_string());
                }
            }
            "reference_text" | "text" | "transcript" => {
                if let Ok(txt) = field.text().await {
                    reference_text = txt.trim().to_string();
                }
            }
            "language" | "lang" => {
                if let Ok(txt) = field.text().await {
                    language = txt.trim().to_string();
                }
            }
            "file" | "audio" | "reference_audio" => {
                if let Ok(bytes) = field.bytes().await {
                    wav_bytes = Some(bytes.to_vec());
                }
            }
            _ => {}
        }
    }

    let Some(voice_name) = name.filter(|s| !s.is_empty()) else {
        let err = warp::reply::with_status(
            warp::reply::json(&ErrorResponse {
                error: "Missing 'name' field in multipart form".to_string(),
            }),
            StatusCode::BAD_REQUEST,
        );
        return Ok(Box::new(err));
    };

    let Some(audio_data) = wav_bytes.filter(|b| !b.is_empty()) else {
        let err = warp::reply::with_status(
            warp::reply::json(&ErrorResponse {
                error: "Missing 'file' (WAV audio) field in multipart form".to_string(),
            }),
            StatusCode::BAD_REQUEST,
        );
        return Ok(Box::new(err));
    };

    let res = tokio::task::spawn_blocking(move || {
        engine.clone_voice(voice_name, reference_text, language, &audio_data)
    })
    .await;

    match res {
        Ok(Ok((prof, deduplicated))) => {
            let reply = CloneVoiceResponse {
                status: "ok".to_string(),
                voice: prof.name.clone(),
                deduplicated,
                message: if deduplicated {
                    format!(
                        "Audio matched existing voice '{}' via SHA-256 deduplication",
                        prof.name
                    )
                } else {
                    format!("Cloned and encoded voice '{}'", prof.name)
                },
            };
            Ok(Box::new(warp::reply::with_status(
                warp::reply::json(&reply),
                StatusCode::OK,
            )))
        }
        Ok(Err(e)) => Ok(Box::new(warp::reply::with_status(
            warp::reply::json(&ErrorResponse {
                error: e.to_string(),
            }),
            StatusCode::BAD_REQUEST,
        ))),
        Err(e) => Ok(Box::new(warp::reply::with_status(
            warp::reply::json(&ErrorResponse {
                error: format!("Worker task panicked: {}", e),
            }),
            StatusCode::INTERNAL_SERVER_ERROR,
        ))),
    }
}

fn urlencoding_decode(input: &str) -> String {
    let mut out = Vec::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(h), Some(l)) = (
                (bytes[i + 1] as char).to_digit(16),
                (bytes[i + 2] as char).to_digit(16),
            ) {
                out.push((h * 16 + l) as u8);
                i += 3;
                continue;
            }
        } else if bytes[i] == b'+' {
            out.push(b' ');
            i += 1;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}
