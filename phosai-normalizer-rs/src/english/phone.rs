//! English Phone Number Verbaliser.

use regex::Regex;
use once_cell::sync::Lazy;

static PHONE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(?:\+\d{1,3}[\s\-]|\+1[\s\-]?|0\d{1,4}[\s\-])\d{2,4}[\s\-]?\d{2,4}(?:[\s\-]?\d{1,4})?\b|\b\d{3,4}[\-]\d{2,4}[\-]\d{2,4}\b").unwrap()
});

fn map_digit(d: char) -> Option<&'static str> {
    match d {
        '0' => Some("zero"),
        '1' => Some("one"),
        '2' => Some("two"),
        '3' => Some("three"),
        '4' => Some("four"),
        '5' => Some("five"),
        '6' => Some("six"),
        '7' => Some("seven"),
        '8' => Some("eight"),
        '9' => Some("nine"),
        '+' => Some("plus"),
        '*' => Some("star"),
        '#' => Some("hash"),
        _ => None,
    }
}

pub fn replace_phone_numbers_en(text: &str) -> String {
    PHONE_RE.replace_all(text, |caps: &regex::Captures| {
        let raw = &caps[0];
        let words: Vec<&'static str> = raw.chars().filter_map(map_digit).collect();
        words.join(" ")
    }).into_owned()
}
