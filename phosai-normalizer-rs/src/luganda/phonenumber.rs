//! Luganda Telephone Number Verbaliser.

use regex::Regex;
use once_cell::sync::Lazy;

static PHONE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(?:\+?256|0)(?:[\s\-]?\d){8,10}\b").unwrap()
});

fn map_phone_digit(d: char) -> Option<&'static str> {
    match d {
        '0' => Some("zeero"),
        '1' => Some("emu"),
        '2' => Some("bbiri"),
        '3' => Some("ssatu"),
        '4' => Some("nnya"),
        '5' => Some("ttaano"),
        '6' => Some("mukaaga"),
        '7' => Some("musanvu"),
        '8' => Some("munaana"),
        '9' => Some("mwenda"),
        '+' => Some("plasi"),
        '*' => Some("sitaba"),
        '#' => Some("hasu"),
        _ => None,
    }
}

pub fn replace_phone_numbers_in_text(text: &str) -> String {
    PHONE_RE.replace_all(text, |caps: &regex::Captures| {
        let raw_phone = &caps[0];
        let words: Vec<&'static str> = raw_phone.chars().filter_map(map_phone_digit).collect();
        words.join(" ")
    }).into_owned()
}
