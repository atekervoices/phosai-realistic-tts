//! Swahili (Kiswahili) Abbreviation and Acronym Expansion.

use regex::Regex;
use once_cell::sync::Lazy;

const ABBREVIATIONS: &[(&str, &str)] = &[
    ("Bw.", "Bwana"),
    ("Bw", "Bwana"),
    ("Bi.", "Bibi"),
    ("Bi", "Bibi"),
    ("Dr.", "Daktari"),
    ("Dr", "Daktari"),
    ("Prof.", "Profesa"),
    ("Prof", "Profesa"),
    ("Mth.", "Mchungaji"),
    ("Mth", "Mchungaji"),
    ("Ust.", "Ustadh"),
    ("Ust", "Ustadh"),
    ("Eng.", "Mhandisi"),
    ("Eng", "Mhandisi"),
    ("Mkt.", "Mwenyekiti"),
    ("mkt.", "mwenyekiti"),
    ("k.m.", "kwa mfano"),
    ("e.g.", "kwa mfano"),
    ("e.g", "kwa mfano"),
    ("i.e.", "yaani"),
    ("i.e", "yaani"),
    ("n.k.", "na kadhalika"),
    ("nk.", "na kadhalika"),
    ("etc.", "na kadhalika"),
    ("etc", "na kadhalika"),
    ("na.", "namba"),
    ("Na.", "Namba"),
    ("S.L.P.", "Sanduku la Posta"),
    ("SLP", "Sanduku la Posta"),
    ("EAC", "Jumuiya ya Afrika Mashariki"),
    ("AU", "Umoja wa Afrika"),
    ("UN", "Umoja wa Mataifa"),
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
