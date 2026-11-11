//! Luganda Diacritics and Accent Canonicalizer.

use unicode_normalization::UnicodeNormalization;

/// Strips accent marks while preserving standard apostrophes.
pub fn normalize_diacritics(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    let nfd: String = text.nfd().collect();
    let cleaned: String = nfd
        .chars()
        .filter(|&c| {
            // Remove non-spacing marks (combining accents) except when needed
            let cat = unicode_normalization::char::canonical_combining_class(c);
            cat == 0
        })
        .collect();
    cleaned.nfc().collect()
}
