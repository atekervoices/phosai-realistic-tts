//! Luganda Elision and Contraction Expander.

use regex::Regex;
use once_cell::sync::Lazy;

const ELISION_PREFIXES: &[(&str, &str)] = &[
    ("n'", "na "),
    ("w'", "wa "),
    ("y'", "ya "),
    ("b'", "ba "),
    ("z'", "za "),
    ("g'", "ga "),
    ("gw'", "gwa "),
    ("bw'", "bwa "),
    ("l'", "la "),
    ("k'", "ka "),
    ("ky'", "kya "),
    ("by'", "bya "),
    ("m'", "mu "),
    ("mw'", "mwa "),
];

static ELISION_PATTERNS: Lazy<Vec<(Regex, &'static str)>> = Lazy::new(|| {
    ELISION_PREFIXES
        .iter()
        .map(|(prefix, expansion)| {
            let pat = format!(r"(^|[^\w]){}(\w+)", regex::escape(prefix));
            (Regex::new(&pat).unwrap(), *expansion)
        })
        .collect()
});

pub fn expand_elisions(mut text: String) -> String {
    for (re, expansion) in ELISION_PATTERNS.iter() {
        text = re.replace_all(&text, |caps: &regex::Captures| {
            let pfx = caps.get(1).map_or("", |m| m.as_str());
            let word = &caps[2];
            format!("{}{}{}", pfx, expansion, word)
        }).into_owned();
    }
    text
}
