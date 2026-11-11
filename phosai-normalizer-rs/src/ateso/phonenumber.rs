//! Ateso Phone Number Verbaliser.

use regex::Regex;
use once_cell::sync::Lazy;

static PHONE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(?:\+?256|0)(?:[\s\-]?\d){8,10}\b").unwrap()
});

fn map_digit(d: char) -> Option<&'static str> {
    match d {
        '0' => Some("jero"),
        '1' => Some("ediopet"),
        '2' => Some("iarei"),
        '3' => Some("iuni"),
        '4' => Some("ioangon"),
        '5' => Some("ikany"),
        '6' => Some("ikanyape"),
        '7' => Some("ikanyaarei"),
        '8' => Some("ikanyauni"),
        '9' => Some("ikanyaongon"),
        '+' => Some("ka"),
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
