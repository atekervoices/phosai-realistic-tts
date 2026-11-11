//! Ateso Ordinal Number & Roman Numeral Verbaliser.

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

static ATE_ORD_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(lo|na|yen)\s+(\d+)\b").unwrap()
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

pub fn ordinal_to_ateso(n: usize) -> String {
    match n {
        1 => "losodit".to_string(),
        2 => "lokiareit".to_string(),
        3 => "louniet".to_string(),
        4 => "lowongonet".to_string(),
        5 => "loikanyet".to_string(),
        10 => "loitomonet".to_string(),
        _ => {
            let word = convert_cardinal_integer(n as i128);
            format!("lo{}et", word)
        }
    }
}

pub fn replace_ordinals_in_text(text: &str) -> String {
    // 1. English style ordinals (1st, 2nd, 3rd, 4th)
    let res = EN_ORD_RE.replace_all(text, |caps: &regex::Captures| {
        let n: usize = caps[1].parse().unwrap_or(1);
        ordinal_to_ateso(n)
    });

    // 2. Ateso prefix + number (e.g. lo 1, lo 2)
    let res = ATE_ORD_RE.replace_all(&res, |caps: &regex::Captures| {
        let n: usize = caps[2].parse().unwrap_or(1);
        ordinal_to_ateso(n)
    });

    // 3. Roman numerals (e.g. Etem II)
    let mut out = res.into_owned();
    for (re, val) in ROMAN_PATTERNS.iter() {
        out = re.replace_all(&out, |caps: &regex::Captures| {
            let name = &caps[1];
            let ord_str = ordinal_to_ateso(*val);
            format!("{} {}", name, ord_str)
        }).into_owned();
    }

    out
}
