//! Luganda Numeric Range Verbaliser.

use regex::Regex;
use once_cell::sync::Lazy;
use super::cardinals::convert_cardinal_integer;

static RANGE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(\d+)\s*-\s*(\d+)\b").unwrap()
});

pub fn replace_ranges_in_text(text: &str) -> String {
    RANGE_RE.replace_all(text, |caps: &regex::Captures| {
        let start: u64 = caps[1].parse().unwrap_or(0);
        let end: u64 = caps[2].parse().unwrap_or(0);
        let start_words = convert_cardinal_integer(start, 0);
        let end_words = convert_cardinal_integer(end, 0);
        format!("{} okutuuka ku {}", start_words, end_words)
    }).into_owned()
}
