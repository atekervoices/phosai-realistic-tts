//! English Titles and Abbreviations Verbaliser.

use regex::Regex;
use once_cell::sync::Lazy;

const ENGLISH_ABBREVIATIONS: &[(&str, &str)] = &[
    ("Dr.", "Doctor"),
    ("Dr", "Doctor"),
    ("Mr.", "Mister"),
    ("Mr", "Mister"),
    ("Mrs.", "Missus"),
    ("Mrs", "Missus"),
    ("Ms.", "Miss"),
    ("Ms", "Miss"),
    ("Prof.", "Professor"),
    ("Prof", "Professor"),
    ("St.", "Saint"),
    ("St", "Saint"),
    ("e.g.", "for example"),
    ("e.g", "for example"),
    ("i.e.", "that is"),
    ("i.e", "that is"),
    ("etc.", "et cetera"),
    ("etc", "et cetera"),
    ("vs.", "versus"),
    ("vs", "versus"),
    ("NASA", "N A S A"),
    ("UNESCO", "U N E S C O"),
    ("WHO", "W H O"),
    ("UN", "U N"),
];

static PATTERNS: Lazy<Vec<(Regex, &'static str)>> = Lazy::new(|| {
    ENGLISH_ABBREVIATIONS
        .iter()
        .map(|(abbrev, expansion)| {
            let pat = format!(r"(^|[^\w]){}([^\w]|$)", regex::escape(abbrev));
            (Regex::new(&pat).unwrap(), *expansion)
        })
        .collect()
});

pub fn replace_abbreviations_en(mut text: String) -> String {
    for (re, expansion) in PATTERNS.iter() {
        text = re.replace_all(&text, |caps: &regex::Captures| {
            let prefix = caps.get(1).map_or("", |m| m.as_str());
            let suffix = caps.get(2).map_or("", |m| m.as_str());
            format!("{}{}{}", prefix, expansion, suffix)
        }).into_owned();
    }
    text
}
