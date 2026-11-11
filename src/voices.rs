use anyhow::{Context, Result};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::audio;
use crate::onnx_engine::Audio8OnnxEngine;

/// Entry in `speaker_registry.json` from `speaker_references_bundle.zip`.
/// Supports both `{"reference_audio": "...", "reference_text": "...", "source_path": "..."}`
/// and extended metadata fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerRegistryEntry {
    #[serde(default, alias = "audio_path")]
    pub reference_audio: String,
    #[serde(default)]
    pub reference_text: String,
    #[serde(default)]
    pub source_path: Option<String>,
    #[serde(default)]
    pub speaker_id: Option<String>,
    #[serde(default)]
    pub language: Option<String>,
}

/// Cached or registered voice entry containing pre-encoded `[num_codebooks, T]` VQ codes,
/// reference transcript, and optional path to raw `.wav` audio.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceProfile {
    pub name: String,
    #[serde(default)]
    pub language: String,
    #[serde(default)]
    pub reference_text: String,
    #[serde(default)]
    pub audio_path: Option<PathBuf>,
    /// Pre-encoded VQ codes of shape `[num_codebooks, num_frames]`.
    #[serde(default)]
    pub vq_codes: Option<Vec<Vec<i64>>>,
    #[serde(default)]
    pub sha256: Option<String>,
}

impl VoiceProfile {
    pub fn codes_array(&self, expected_codebooks: usize) -> Option<Array2<i64>> {
        let codes = self.vq_codes.as_ref()?;
        if codes.len() != expected_codebooks || codes[0].is_empty() {
            return None;
        }
        let cols = codes[0].len();
        let mut arr = Array2::<i64>::zeros((expected_codebooks, cols));
        for (r, row) in codes.iter().enumerate() {
            for (c, &val) in row.iter().take(cols).enumerate() {
                arr[[r, c]] = val;
            }
        }
        Some(arr)
    }

    pub fn set_codes_from_array(&mut self, arr: &Array2<i64>) {
        let (rows, cols) = arr.dim();
        let mut nested = Vec::with_capacity(rows);
        for r in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for c in 0..cols {
                row.push(arr[[r, c]]);
            }
            nested.push(row);
        }
        self.vq_codes = Some(nested);
    }
}

pub struct VoiceRegistry {
    pub voices_dir: PathBuf,
    pub registry_json_path: PathBuf,
    pub profiles: HashMap<String, VoiceProfile>,
    /// Maps short alias (e.g. `lug_female_10`) -> canonical key (e.g. `crestai_vo_tts:lug_female_10`).
    pub aliases: HashMap<String, String>,
    /// Maps SHA-256 hex digest of raw audio bytes -> canonical voice key.
    pub hash_to_voice: HashMap<String, String>,
}

impl VoiceRegistry {
    pub fn load(
        voices_dir: impl AsRef<Path>,
        registry_json_path: impl AsRef<Path>,
        reference_root: impl AsRef<Path>,
    ) -> Result<Self> {
        let voices_dir = voices_dir.as_ref().to_path_buf();
        let registry_json_path = registry_json_path.as_ref().to_path_buf();
        let reference_root = reference_root.as_ref().to_path_buf();

        if !voices_dir.exists() {
            let _ = fs::create_dir_all(&voices_dir);
        }

        let mut registry = Self {
            voices_dir,
            registry_json_path: registry_json_path.clone(),
            profiles: HashMap::new(),
            aliases: HashMap::new(),
            hash_to_voice: HashMap::new(),
        };

        // 1. Load bundled `speaker_registry.json` (94 speakers from `speaker_references_bundle.zip`)
        if registry_json_path.is_file() {
            let content = fs::read_to_string(&registry_json_path).with_context(|| {
                format!(
                    "Failed to read speaker registry at {}",
                    registry_json_path.display()
                )
            })?;
            let map: HashMap<String, SpeakerRegistryEntry> = serde_json::from_str(&content)
                .with_context(|| {
                    format!(
                        "Failed to parse speaker_registry.json at {}",
                        registry_json_path.display()
                    )
                })?;

            for (canonical_key, entry) in map {
                let wav_path = reference_root.join(&entry.reference_audio);
                let short_id = entry
                    .speaker_id
                    .clone()
                    .unwrap_or_else(|| {
                        canonical_key
                            .split(':')
                            .nth(1)
                            .unwrap_or(&canonical_key)
                            .to_string()
                    });
                let language = entry.language.clone().unwrap_or_else(|| {
                    phosai_normalizer::detect_language_from_speaker(&short_id).to_string()
                });

                let safe_stem = canonical_key.replace(':', "__");
                let cached_json = registry.voices_dir.join(format!("{}.json", safe_stem));
                let mut vq_codes = None;
                let mut sha256 = None;

                if cached_json.is_file() {
                    if let Ok(cached_str) = fs::read_to_string(&cached_json) {
                        if let Ok(cached_prof) = serde_json::from_str::<VoiceProfile>(&cached_str) {
                            vq_codes = cached_prof.vq_codes;
                            sha256 = cached_prof.sha256;
                        }
                    }
                }

                let profile = VoiceProfile {
                    name: canonical_key.clone(),
                    language,
                    reference_text: entry.reference_text.clone(),
                    audio_path: Some(wav_path),
                    vq_codes,
                    sha256: sha256.clone(),
                };

                if let Some(h) = sha256 {
                    registry.hash_to_voice.insert(h, canonical_key.clone());
                }

                let should_override = match registry.aliases.get(&short_id) {
                    None => true,
                    Some(existing) => {
                        (canonical_key.contains("salt") || canonical_key.contains("crestai"))
                            && !existing.contains("salt")
                            && !existing.contains("crestai")
                    }
                };
                if should_override {
                    registry.aliases.insert(short_id, canonical_key.clone());
                }

                registry.profiles.insert(canonical_key, profile);
            }
        }

        // 2. Also load any custom `.json` voice profiles saved in `voices_dir`
        if let Ok(entries) = fs::read_dir(&registry.voices_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("json") {
                    if let Ok(text) = fs::read_to_string(&path) {
                        if let Ok(prof) = serde_json::from_str::<VoiceProfile>(&text) {
                            if let Some(ref h) = prof.sha256 {
                                registry.hash_to_voice.insert(h.clone(), prof.name.clone());
                            }
                            registry
                                .profiles
                                .entry(prof.name.clone())
                                .and_modify(|existing| {
                                    if existing.vq_codes.is_none() && prof.vq_codes.is_some() {
                                        existing.vq_codes = prof.vq_codes.clone();
                                    }
                                    if !prof.reference_text.is_empty() {
                                        existing.reference_text = prof.reference_text.clone();
                                    }
                                })
                                .or_insert(prof);
                        }
                    }
                }
            }
        }

        println!(
            "VoiceRegistry loaded {} speaker profiles ({} short aliases).",
            registry.profiles.len(),
            registry.aliases.len()
        );

        Ok(registry)
    }

    pub fn resolve_key(&self, requested: &str) -> Option<String> {
        let trimmed = requested.trim();
        if trimmed.is_empty() {
            return None;
        }
        if self.profiles.contains_key(trimmed) {
            return Some(trimmed.to_string());
        }
        if let Some(canon) = self.aliases.get(trimmed) {
            return Some(canon.clone());
        }
        let lower = trimmed.to_lowercase();
        for k in self.profiles.keys() {
            if k.to_lowercase() == lower {
                return Some(k.clone());
            }
        }
        for (alias, canon) in &self.aliases {
            if alias.to_lowercase() == lower {
                return Some(canon.clone());
            }
        }
        if let Some((lang_gender, _num)) = lower.rsplit_once('_') {
            if lang_gender.contains('_') {
                let mut candidates: Vec<&String> = self
                    .profiles
                    .keys()
                    .filter(|k| {
                        let short = k.split(':').nth(1).unwrap_or(k).to_lowercase();
                        short.starts_with(&format!("{}_", lang_gender))
                    })
                    .collect();
                candidates.sort();
                if let Some(first) = candidates.first() {
                    return Some((*first).clone());
                }
            }
        }
        None
    }

    pub fn list_voices(&self) -> Vec<String> {
        let mut names: Vec<String> = self.profiles.keys().cloned().collect();
        names.sort();
        names
    }

    pub fn get_profile(&self, name: &str) -> Option<&VoiceProfile> {
        let key = self.resolve_key(name)?;
        self.profiles.get(&key)
    }

    pub fn ensure_encoded_profile(
        &mut self,
        name: &str,
        engine: &mut Audio8OnnxEngine,
    ) -> Result<VoiceProfile> {
        let key = self
            .resolve_key(name)
            .ok_or_else(|| anyhow::anyhow!("Voice '{}' not found in registry", name))?;

        let num_cb = engine.manifest.num_codebooks;
        if let Some(existing) = self.profiles.get(&key) {
            if existing.codes_array(num_cb).is_some() {
                return Ok(existing.clone());
            }
        }

        let profile = self
            .profiles
            .get(&key)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Voice '{}' missing from registry", key))?;

        let wav_path = profile.audio_path.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "Voice '{}' has neither cached VQ codes nor an audio_path on disk",
                key
            )
        })?;

        if !wav_path.is_file() {
            anyhow::bail!(
                "Reference audio file for voice '{}' not found at {}",
                key,
                wav_path.display()
            );
        }

        let samples_44k =
            audio::load_audio_mono_resampled(wav_path, engine.manifest.sample_rate)?;
        let codes_arr = engine.encode_reference_audio(&samples_44k)?;

        let mut updated = profile;
        updated.set_codes_from_array(&codes_arr);

        let safe_stem = key.replace(':', "__");
        let cache_file = self.voices_dir.join(format!("{}.json", safe_stem));
        if let Ok(serialized) = serde_json::to_string(&updated) {
            let _ = fs::write(&cache_file, serialized);
        }

        self.profiles.insert(key, updated.clone());
        Ok(updated)
    }

    pub fn register_cloned_voice(
        &mut self,
        name: String,
        reference_text: String,
        language: String,
        wav_bytes: &[u8],
        engine: &mut Audio8OnnxEngine,
    ) -> Result<(VoiceProfile, bool)> {
        let mut hasher = Sha256::new();
        hasher.update(wav_bytes);
        let hash_hex = hex::encode(hasher.finalize());

        if let Some(existing_name) = self.hash_to_voice.get(&hash_hex).cloned() {
            if let Some(existing_prof) = self.profiles.get(&existing_name).cloned() {
                return Ok((existing_prof, true));
            }
        }

        let samples_44k =
            audio::decode_wav_bytes_resampled(wav_bytes, engine.manifest.sample_rate)?;
        let codes_arr = engine.encode_reference_audio(&samples_44k)?;

        let safe_stem = name.replace(':', "__");
        let wav_out_path = self.voices_dir.join(format!("{}.wav", safe_stem));
        let _ = fs::write(&wav_out_path, wav_bytes);

        let mut profile = VoiceProfile {
            name: name.clone(),
            language,
            reference_text,
            audio_path: Some(wav_out_path),
            vq_codes: None,
            sha256: Some(hash_hex.clone()),
        };
        profile.set_codes_from_array(&codes_arr);

        let json_out_path = self.voices_dir.join(format!("{}.json", safe_stem));
        let serialized = serde_json::to_string(&profile)?;
        fs::write(&json_out_path, serialized)?;

        self.hash_to_voice.insert(hash_hex, name.clone());
        self.aliases.insert(name.clone(), name.clone());
        self.profiles.insert(name, profile.clone());

        Ok((profile, false))
    }

    pub fn delete_voice(&mut self, name: &str) -> Result<()> {
        let key = self
            .resolve_key(name)
            .ok_or_else(|| anyhow::anyhow!("Voice '{}' not found", name))?;

        if let Some(prof) = self.profiles.remove(&key) {
            if let Some(h) = prof.sha256 {
                self.hash_to_voice.remove(&h);
            }
        }
        self.aliases.retain(|_, v| v != &key);

        let safe_stem = key.replace(':', "__");
        let json_path = self.voices_dir.join(format!("{}.json", safe_stem));
        let wav_path = self.voices_dir.join(format!("{}.wav", safe_stem));
        if json_path.is_file() {
            let _ = fs::remove_file(json_path);
        }
        if wav_path.is_file() {
            let _ = fs::remove_file(wav_path);
        }
        Ok(())
    }
}
