//! Kinyarwanda Abbreviation and Acronym Expansion.

use regex::Regex;
use once_cell::sync::Lazy;

const ABBREVIATIONS: &[(&str, &str)] = &[
    ("Bw.", "Bwana"),
    ("Bw", "Bwana"),
    ("Bi.", "Bibi"),
    ("Bi", "Bibi"),
    ("Dr.", "dogiteri"),
    ("Dr", "dogiteri"),
    ("Prof.", "purofu"),
    ("Prof", "purofu"),
    ("RBC", "arabisi"),
    ("RISA", "risa"),
    ("RDB", "aradibi"),
    ("RDF", "aradiyefu"),
    ("OMS", "o e mesi"),
    ("WHO", "o e mesi"),
    ("UN", "oni"),
    ("UNICEF", "yunicefu"),
    ("UNESCO", "yunesico"),
    ("CDC", "sidisi"),
    ("HIV", "heci ayivi"),
    ("BBC", "bibisi"),
    ("VOA", "vi o aye"),
    ("COVID", "kovide"),
    ("PCR", "pisiyara"),
    ("ICU", "ayisiyu"),
    ("FMI", "efu emi"),
    ("HCR", "hashiseyeri"),
    ("k.m.", "nkurugero"),
    ("e.g.", "nkurugero"),
    ("e.g", "nkurugero"),
    ("i.e.", "nukuri"),
    ("i.e", "nukuri"),
    ("n.k.", "n'ibindi"),
    ("etc.", "n'ibindi"),
    ("etc", "n'ibindi"),
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
