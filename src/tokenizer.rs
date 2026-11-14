use anyhow::Result;
use ndarray::{s, Array2, Array3};
use regex::Regex;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tokenizers::Tokenizer;

use crate::config::RuntimeManifest;

const COMMON_ABBREVIATIONS: &[&str] = &[
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "vs.", "e.g.", "i.e.", "etc.", "st.",
    "inc.", "ltd.", "co.", "mw.", "owek.", "omuk.",
];

fn is_abbreviation_boundary(text: &str) -> bool {
    if let Some(last_word) = text.trim().split_whitespace().last() {
        let lw = last_word.to_lowercase();
        COMMON_ABBREVIATIONS.contains(&lw.as_str())
    } else {
        false
    }
}

/// Heuristic language detector from text content, ported directly from `spark-tts-rs`.
pub fn detect_language_from_text(text: &str, fallback_speaker_lang: &str) -> &'static str {
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower
        .split(|c: char| !c.is_alphabetic())
        .filter(|w| !w.is_empty())
        .collect();

    if words.is_empty() {
        return match fallback_speaker_lang {
            "lug" => "lug",
            "swa" => "swa",
            "kin" => "kin",
            "ach" => "ach",
            "teo" => "teo",
            _ => "eng",
        };
    }

    let mut eng_score = 0usize;
    let mut lug_score = 0usize;
    let mut swa_score = 0usize;
    let mut ach_score = 0usize;
    let mut kin_score = 0usize;

    for &w in &words {
        match w {
            "the" | "and" | "to" | "in" | "of" | "is" | "you" | "that" | "it" | "he" | "was"
            | "for" | "on" | "are" | "as" | "with" | "his" | "they" | "at" | "be" | "this"
            | "from" | "have" | "has" | "had" | "or" | "by" | "one" | "not" | "but" | "what"
            | "all" | "were" | "when" | "we" | "there" | "can" | "an" | "your" | "which"
            | "their" | "if" | "will" | "do" | "hello" | "welcome" | "today" | "please"
            | "thank" | "thanks" | "our" | "my" | "who" | "whom" | "about" | "more" | "some"
            | "into" | "out" | "up" | "so" | "them" | "her" | "would" | "make" | "like"
            | "time" | "no" | "just" | "him" | "know" | "take" | "people" | "year" | "years"
            | "good" | "bad" | "could" | "see" | "other" | "than" | "then" | "now" | "look"
            | "only" | "come" | "its" | "over" | "think" | "also" | "back" | "after" | "use"
            | "two" | "how" | "work" | "first" | "well" | "way" | "even" | "new" | "want"
            | "because" | "any" | "these" | "give" | "day" | "most" | "us" => {
                eng_score += 1;
            }
            "era" | "naye" | "kubanga" | "bwe" | "nga" | "ate" | "buli" | "abantu" | "omuntu"
            | "waliwo" | "waffe" | "kyokka" | "oluvanyuma" | "oli" | "otya" | "leero" | "enkya"
            | "eggwanga" | "entebe" | "obulamu" | "ssebo" | "nnyabo" | "ebintu" | "omulimu"
            | "katonda" | "gavumenti" | "nnyo" | "ffe" | "gwe" | "ye" | "bo" | "ebigambo"
            | "amagezi" | "essomero" | "omwana" | "abaana" | "gyebaleko" | "webale" | "ensonga" => {
                lug_score += 1;
            }
            "ya" | "wa" | "kwa" | "katika" | "ni" | "za" | "la" | "cha" | "vya" | "habari"
            | "jambo" | "sana" | "yake" | "watu" | "mungu" | "serikali" | "lakini" | "kama"
            | "hata" | "sasa" | "leo" | "hapa" | "ndiyo" | "hapana" | "nzuri" | "asante"
            | "karibu" | "yangu" | "yetu" | "kazi" | "nchi" | "mambo" => {
                swa_score += 1;
            }
            "kede" | "ento" | "pien" | "ka" | "kit" | "dano" | "pol" | "kare" | "tin" | "kop"
            | "lok" | "rwot" | "woko" | "maber" | "apwoyo" | "latin" | "lotino" => {
                ach_score += 1;
            }
            "kandi" | "ariko" | "kuko" | "muri" | "ku" | "yuko" | "cyane" | "bantu" | "umuntu"
            | "none" | "ejo" | "mwiriwe" | "muraho" | "amakuru" | "yego" | "oya" => {
                kin_score += 1;
            }
            _ => {}
        }
    }

    let max_score = eng_score
        .max(lug_score)
        .max(swa_score)
        .max(ach_score)
        .max(kin_score);
    if max_score >= 2 || (max_score == 1 && words.len() <= 5) {
        if eng_score == max_score {
            return "eng";
        } else if lug_score == max_score {
            return "lug";
        } else if swa_score == max_score {
            return "swa";
        } else if ach_score == max_score {
            return "ach";
        } else if kin_score == max_score {
            return "kin";
        }
    }

    match fallback_speaker_lang {
        "lug" => "lug",
        "swa" => "swa",
        "kin" => "kin",
        "ach" => "ach",
        "teo" => "teo",
        _ => "eng",
    }
}

/// Split text strictly into complete sentences, ported from `spark-tts-rs`.
pub fn chunk_sentences(text: &str) -> Vec<String> {
    const MIN_SENTENCE_CHARS: usize = 6;
    let mut sentences = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();

    let mut i = 0;
    while i < len {
        let c = chars[i];

        if c == '\n' || c == '\r' {
            if i + 1 < len && (chars[i + 1] == '\n' || chars[i + 1] == '\r') {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    sentences.push(trimmed);
                }
                current.clear();
                while i < len && (chars[i] == '\n' || chars[i] == '\r' || chars[i] == ' ') {
                    i += 1;
                }
                continue;
            } else {
                if !current.is_empty() && !current.ends_with(' ') {
                    current.push(' ');
                }
                i += 1;
                continue;
            }
        }

        current.push(c);

        let is_terminal = c == '.' || c == '!' || c == '?';
        if is_terminal {
            let is_decimal = (c == '.')
                && i > 0
                && chars[i - 1].is_ascii_digit()
                && (i + 1 < len && chars[i + 1].is_ascii_digit());

            let is_abbrev = (c == '.') && is_abbreviation_boundary(&current);

            let is_end_of_clause = (i + 1 == len)
                || (i + 1 < len
                    && (chars[i + 1].is_whitespace()
                        || chars[i + 1] == '"'
                        || chars[i + 1] == '\''));

            let trimmed = current.trim();
            let is_list_number =
                (c == '.') && trimmed.chars().all(|ch| ch.is_ascii_digit() || ch == '.');
            let is_number_word = (c == '.')
                && matches!(
                    trimmed.trim_end_matches('.'),
                    "one"
                        | "two"
                        | "three"
                        | "four"
                        | "five"
                        | "six"
                        | "seven"
                        | "eight"
                        | "nine"
                        | "ten"
                );
            let is_list_prefix = is_list_number || is_number_word;

            if !is_decimal
                && !is_abbrev
                && !is_list_prefix
                && is_end_of_clause
                && current.trim().chars().count() >= MIN_SENTENCE_CHARS
            {
                while i + 1 < len
                    && (chars[i + 1] == '"'
                        || chars[i + 1] == '\''
                        || chars[i + 1] == ')'
                        || chars[i + 1] == '»'
                        || chars[i + 1] == '”')
                {
                    i += 1;
                    current.push(chars[i]);
                }
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    sentences.push(trimmed);
                }
                current.clear();
            }
        }

        i += 1;
    }

    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }

    sentences
}

/// Normalize and chunk text using `phosai-normalizer` (matching `spark-tts-rs`).
pub fn normalize_and_chunk(text: &str, lang: &str) -> Vec<String> {
    let pre_cleaned = phosai_normalizer::cleaner::fix_glued_punctuation(text);
    let normalized = phosai_normalizer::normalize(&pre_cleaned, lang);
    let post_cleaned = phosai_normalizer::cleaner::sanitize_tts_post(&normalized);
    let raw_chunks = chunk_sentences(&post_cleaned);
    let chunks = if raw_chunks.is_empty() {
        let t = post_cleaned.trim().to_string();
        if t.is_empty() {
            Vec::new()
        } else {
            vec![t]
        }
    } else {
        raw_chunks
    };

    let mut merged: Vec<String> = Vec::new();
    for c in chunks {
        let trimmed = c.trim().to_string();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(last) = merged.last_mut() {
            let last_trimmed = last.trim();
            let is_digit_bullet = last_trimmed.chars().all(|ch| ch.is_ascii_digit() || ch == '.');
            if is_digit_bullet && last_trimmed.len() <= 4 {
                last.push(' ');
                last.push_str(&trimmed);
                continue;
            }
        }
        merged.push(trimmed);
    }
    merged
}

/// Text normalizer + HuggingFace tokenizer + Audio8 DualAR `[1, num_codebooks + 1, seq_len]` prompt builder.
/// Tokenizes text prompts and acoustic code matrices matching `spark-tts-rs`.
pub struct Audio8PromptTokenizer {
    tokenizer: Tokenizer,
    special_tokens: HashMap<String, i64>,
    special_regex: Option<Regex>,
}

impl Audio8PromptTokenizer {
    pub fn from_model_dir(model_dir: &Path, manifest: &RuntimeManifest) -> Result<Self> {
        let primary = model_dir.join("tokenizer.json");
        let nested = model_dir.join("tokenizer").join("tokenizer.json");
        let tok_path = if primary.is_file() {
            primary
        } else if nested.is_file() {
            nested
        } else {
            anyhow::bail!(
                "tokenizer.json not found in {} (run `python scripts/export_torchscript.py` first to export the model into `./model`)",
                model_dir.display()
            );
        };

        let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| {
            anyhow::anyhow!("Failed to load tokenizer from {}: {}", tok_path.display(), e)
        })?;

        let mut special_tokens = HashMap::new();
        special_tokens.insert("<|im_start|>".to_string(), 151644);
        special_tokens.insert("<|im_end|>".to_string(), manifest.im_end_id);
        special_tokens.insert("<|voice|>".to_string(), 151673);
        special_tokens.insert("<|text|>".to_string(), 151672);

        let tok_cfg_path = model_dir.join("tokenizer_config.json");
        if tok_cfg_path.is_file() {
            if let Ok(raw) = fs::read_to_string(&tok_cfg_path) {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&raw) {
                    if let Some(added) = val.get("added_tokens_decoder").and_then(|v| v.as_object())
                    {
                        for (id_str, info) in added {
                            if let (Ok(id), Some(content)) = (
                                id_str.parse::<i64>(),
                                info.get("content").and_then(|c| c.as_str()),
                            ) {
                                if !content.starts_with("<|semantic:") {
                                    special_tokens.insert(content.to_string(), id);
                                }
                            }
                        }
                    }
                }
            }
        }

        for (tok, id) in tokenizer.get_vocab(true) {
            if tok.starts_with("<|") && tok.ends_with("|>") && !tok.starts_with("<|semantic:") {
                special_tokens.insert(tok, id as i64);
            }
        }

        let mut keys: Vec<String> = special_tokens
            .keys()
            .map(|k| regex::escape(k))
            .collect();
        keys.sort_by_key(|k| std::cmp::Reverse(k.len()));
        let special_regex = if keys.is_empty() {
            None
        } else {
            Regex::new(&format!("({})", keys.join("|"))).ok()
        };

        Ok(Self {
            tokenizer,
            special_tokens,
            special_regex,
        })
    }

    /// Normalize and split input text into sentence chunks using `phosai-normalizer`.
    pub fn normalize_and_split(&self, text: &str, speaker_or_lang_hint: Option<&str>) -> Vec<String> {
        let fallback_lang = speaker_or_lang_hint
            .map(|s| {
                let short = s.split(':').nth(1).unwrap_or(s);
                phosai_normalizer::detect_language_from_speaker(short)
            })
            .unwrap_or("eng");
        let detected_lang = detect_language_from_text(text, fallback_lang);
        normalize_and_chunk(text, detected_lang)
    }

    /// Encode text while preserving special tokens like `<|im_start|>`, `<|im_end|>`, `<|voice|>`.
    pub fn encode_with_specials(&self, text: &str) -> Result<Vec<i64>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        let Some(ref re) = self.special_regex else {
            let enc = self
                .tokenizer
                .encode(text, false)
                .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
            return Ok(enc.get_ids().iter().map(|&id| id as i64).collect());
        };

        let mut out = Vec::new();
        let mut last_end = 0;
        for mat in re.find_iter(text) {
            if mat.start() > last_end {
                let sub = &text[last_end..mat.start()];
                let enc = self
                    .tokenizer
                    .encode(sub, false)
                    .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
                out.extend(enc.get_ids().iter().map(|&id| id as i64));
            }
            let sym = mat.as_str();
            if let Some(&id) = self.special_tokens.get(sym) {
                out.push(id);
            }
            last_end = mat.end();
        }
        if last_end < text.len() {
            let tail = &text[last_end..];
            let enc = self
                .tokenizer
                .encode(tail, false)
                .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
            out.extend(enc.get_ids().iter().map(|&id| id as i64));
        }

        Ok(out)
    }

    fn text_column(&self, tokens: &[i64], num_codebooks: usize) -> Array2<i64> {
        let mut arr = Array2::<i64>::zeros((num_codebooks + 1, tokens.len()));
        for (col, &tok) in tokens.iter().enumerate() {
            arr[[0, col]] = tok;
        }
        arr
    }

    /// Build the `[1, num_codebooks + 1, seq_len]` DualAR prompt tensor matching
    /// `ArkttsProcessor` and `modeling_arktts.py` in `phosai/realistic-v1`.
    pub fn build_prompt(
        &self,
        target_text: &str,
        reference_codes: Option<&Array2<i64>>,
        reference_text: Option<&str>,
        _speaker_tag: Option<&str>,
        manifest: &RuntimeManifest,
    ) -> Result<Array3<i64>> {
        let num_cb = manifest.num_codebooks;
        let cb_size = manifest.codebook_size as i64;
        let sem_begin = manifest.semantic_begin_id;

        if let Some(codes) = reference_codes {
            let (rows, cols) = codes.dim();
            if rows != num_cb {
                anyhow::bail!(
                    "Reference VQ codes have {} rows, expected {}",
                    rows,
                    num_cb
                );
            }
            let raw_ref = reference_text.unwrap_or("").trim();
            let ref_txt = if raw_ref.contains("<|speaker:") {
                raw_ref.to_string()
            } else {
                format!("<|speaker:0|>{}", raw_ref)
            };

            let prefix_str = format!(
                "<|im_start|>system\nconvert the provided text to speech reference to the following:\n\nText:\n{}\n\nSpeech:\n",
                ref_txt
            );
            let suffix_str = format!(
                "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<|voice|>",
                target_text.trim()
            );

            let prefix_tokens = self.encode_with_specials(&prefix_str)?;
            let suffix_tokens = self.encode_with_specials(&suffix_str)?;

            let total_len = prefix_tokens.len() + cols + suffix_tokens.len();
            let mut prompt = Array3::<i64>::zeros((1, num_cb + 1, total_len));

            // 1. Prefix on row 0
            for (c, &tok) in prefix_tokens.iter().enumerate() {
                prompt[[0, 0, c]] = tok;
            }

            // 2. Reference audio codes in the middle
            let code_offset = prefix_tokens.len();
            for c in 0..cols {
                let first_code = codes[[0, c]].clamp(0, cb_size - 1);
                prompt[[0, 0, code_offset + c]] = sem_begin + first_code;
                for r in 0..num_cb {
                    prompt[[0, r + 1, code_offset + c]] = codes[[r, c]];
                }
            }

            // 3. Suffix on row 0
            let suffix_offset = code_offset + cols;
            for (c, &tok) in suffix_tokens.iter().enumerate() {
                prompt[[0, 0, suffix_offset + c]] = tok;
            }

            Ok(prompt)
        } else {
            let prompt_str = format!(
                "<|im_start|>system\nconvert the provided text to speech<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<|voice|>",
                target_text.trim()
            );
            let tokens = self.encode_with_specials(&prompt_str)?;
            let mut prompt = Array3::<i64>::zeros((1, num_cb + 1, tokens.len()));
            for (c, &tok) in tokens.iter().enumerate() {
                prompt[[0, 0, c]] = tok;
            }
            Ok(prompt)
        }
    }
}
