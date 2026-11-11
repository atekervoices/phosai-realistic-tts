//! Kinyarwanda Phone Number Verbaliser.

use regex::Regex;
use once_cell::sync::Lazy;

static PHONE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(?:\+\d{1,3}[\s\-]?)?(?:07[82389][\s\-]?\d{3}[\s\-]?\d{4}|0[1-9]\d{7,8})\b").unwrap()
});

fn map_digit(d: char) -> Option<&'static str> {
    match d {
        '0' => Some("zero"),
        '1' => Some("rimwe"),
        '2' => Some("kabiri"),
        '3' => Some("gatatu"),
        '4' => Some("kane"),
        '5' => Some("gatanu"),
        '6' => Some("gatandatu"),
        '7' => Some("karindwi"),
        '8' => Some("umunani"),
        '9' => Some("icyenda"),
        '+' => Some("terana"),
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
