//! Acholi Ordinal Number & Roman Numeral Verbaliser.

use regex::Regex;
use once_cell::sync::Lazy;
use super::cardinals::convert_cardinal_integer;

const ROMAN_MAP: &[(&str, usize)] = &[
    ("XV", 15), ("XIV", 14), ("XIII", 13), ("XII", 12), ("XI", 11),
    ("X", 10), ("IX", 9), ("VIII", 8), ("VII", 7), ("VI", 6),
    ("V", 5), ("IV", 4), ("III", 3), ("II", 2), ("I", 1),
];

static EN_ORD_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(\d+)(?:st|nd|rd|th)\b").unwrap()
});

static ACH_ORD_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(me)\s+(\d+)\b").unwrap()
});

static ROMAN_PATTERNS: Lazy<Vec<(Regex, usize)>> = Lazy::new(|| {
    ROMAN_MAP
        .iter()
        .map(|(rom, val)| {
            let pat = format!(r"\b([A-Z][a-zA-Z]+)\s+{}\b", regex::escape(rom));
            (Regex::new(&pat).unwrap(), *val)
        })
        .collect()
});

pub fn ordinal_to_acholi(n: usize, prefix: &str) -> String {
    let word = convert_cardinal_integer(n as i128);
    format!("{} {}", prefix, word)
}

pub fn replace_ordinals_in_text(text: &str) -> String {
    // 1. English style ordinals (1st, 2nd, 3rd, 4th)
    let res = EN_ORD_RE.replace_all(text, |caps: &regex::Captures| {
        let n: usize = caps[1].parse().unwrap_or(1);
        ordinal_to_acholi(n, "Me")
    });

    // 2. Acholi prefix + number (e.g. Me 1, Me 2)
    let res = ACH_ORD_RE.replace_all(&res, |caps: &regex::Captures| {
        let n: usize = caps[2].parse().unwrap_or(1);
        ordinal_to_acholi(n, "Me")
    });

    // 3. Roman numerals (e.g. Okello II)
    let mut out = res.into_owned();
    for (re, val) in ROMAN_PATTERNS.iter() {
        out = re.replace_all(&out, |caps: &regex::Captures| {
            let name = &caps[1];
            let ord_str = ordinal_to_acholi(*val, "Me");
            format!("{} {}", name, ord_str)
        }).into_owned();
    }

    out
}
