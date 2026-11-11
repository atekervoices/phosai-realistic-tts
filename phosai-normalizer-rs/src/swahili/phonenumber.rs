//! Swahili (Kiswahili) Phone Number Verbaliser.

use regex::Regex;
use once_cell::sync::Lazy;

static PHONE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(?:\+\d{1,3}[\s\-]?)?(?:0\d{2,4}[\s\-]?\d{3,4}[\s\-]?\d{3,4}|0[167]\d{8})\b").unwrap()
});

fn map_digit(d: char) -> Option<&'static str> {
    match d {
        '0' => Some("sifuri"),
        '1' => Some("moja"),
        '2' => Some("mbili"),
        '3' => Some("tatu"),
        '4' => Some("nne"),
        '5' => Some("tano"),
        '6' => Some("sita"),
        '7' => Some("saba"),
        '8' => Some("nane"),
        '9' => Some("tisa"),
        '+' => Some("jumlisha"),
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
