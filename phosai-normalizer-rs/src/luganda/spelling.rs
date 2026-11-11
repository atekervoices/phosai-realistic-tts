//! Luganda Spelling Standardization & Typo Correction Engine.

use regex::Regex;
use once_cell::sync::Lazy;
use std::collections::HashMap;

static WORD_CLEAN_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[^\w]").unwrap());

// Common spelling standardizations
const SPELLING_MAPPINGS: &[(&str, &str)] = &[
    ("abantu", "abantu"),
    ("ddala", "ddala"),
    ("nnyo", "nnyo"),
    ("bulungi", "bulungi"),
    ("mpozzi", "mpozzi"),
    ("singa", "singa"),
];

static SPELLING_MAP: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    let mut map = HashMap::new();
    for (k, v) in SPELLING_MAPPINGS {
        map.insert(*k, *v);
    }
    map
});

pub fn standardize_spelling(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut tokens: Vec<String> = words.into_iter().map(|w| w.to_string()).collect();

    for token in tokens.iter_mut() {
        let clean_token = WORD_CLEAN_RE.replace_all(token, "").to_lowercase();
        if let Some(&canonical) = SPELLING_MAP.get(clean_token.as_str()) {
            let is_first_upper = token.chars().next().map_or(false, |c| c.is_uppercase());
            let replacement = if is_first_upper {
                let mut c = canonical.chars();
                match c.next() {
                    None => String::new(),
                    Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                }
            } else {
                canonical.to_string()
            };
            *token = token.replace(&clean_token, &replacement);
        }
    }

    tokens.join(" ")
}
