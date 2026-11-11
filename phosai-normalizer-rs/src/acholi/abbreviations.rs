//! Acholi Abbreviation and Acronym Expansion.

use regex::Regex;
use once_cell::sync::Lazy;

const ABBREVIATIONS: &[(&str, &str)] = &[
    ("Dr.", "Dokta"),
    ("Dr", "Dokta"),
    ("Prof.", "Propesa"),
    ("Prof", "Propesa"),
    ("e.g.", "labole"),
    ("e.g", "labole"),
    ("i.e.", "teloke"),
    ("i.e", "teloke"),
    ("etc.", "ki mukene"),
    ("etc", "ki mukene"),
    ("URA", "Uganda Revenue Authority"),
    ("UNRA", "Uganda National Roads Authority"),
    ("NSSF", "National Social Security Fund"),
];

static PATTERNS: Lazy<Vec<(Regex, &'static str)>> = Lazy::new(|| {
    ABBREVIATIONS
        .iter()
        .map(|(abbrev, expansion)| {
            let pat = format!(r"(^|[^\w]){}([^\w]|$)", regex::escape(abbrev));
            (Regex::new(&pat).unwrap(), *expansion)
        })
        .collect()
});

pub fn replace_abbreviations_in_text(mut text: String) -> String {
    for (re, expansion) in PATTERNS.iter() {
        text = re.replace_all(&text, |caps: &regex::Captures| {
            let prefix = caps.get(1).map_or("", |m| m.as_str());
            let suffix = caps.get(2).map_or("", |m| m.as_str());
            format!("{}{}{}", prefix, expansion, suffix)
        }).into_owned();
    }
    text
}
