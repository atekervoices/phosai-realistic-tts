//! Acholi Phone Number Verbaliser.

use regex::Regex;
use once_cell::sync::Lazy;

static PHONE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(?:\+?256|0)(?:[\s\-]?\d){8,10}\b").unwrap()
});

fn map_digit(d: char) -> Option<&'static str> {
    match d {
        '0' => Some("jero"),
        '1' => Some("acel"),
        '2' => Some("aryo"),
        '3' => Some("adek"),
        '4' => Some("angwen"),
        '5' => Some("abic"),
        '6' => Some("abicel"),
        '7' => Some("abiro"),
        '8' => Some("aboro"),
        '9' => Some("abungwen"),
        '+' => Some("medo"),
        _ => None,
    }
}

pub fn replace_phone_numbers_in_text(text: &str) -> String {
    PHONE_RE.replace_all(text, |caps: &regex::Captures| {
        let phone_str = &caps[0];
        let words: Vec<&'static str> = phone_str.chars().filter_map(map_digit).collect();
        words.join(" ")
    }).into_owned()
}
