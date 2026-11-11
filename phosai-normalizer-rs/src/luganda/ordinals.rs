//! Luganda Ordinal Verbaliser.

use regex::Regex;
use once_cell::sync::Lazy;

const ORDINALS_MAP: [&str; 11] = [
    "ow'okumu",
    "ow'oluberyeberye",
    "ow'okubiri",
    "ow'okusatu",
    "ow'okuna",
    "ow'okutaano",
    "ow'omukaaga",
    "ow'omusanvu",
    "ow'omunaana",
    "ow'omwenda",
    "ow'ekkumi",
];

const EKYASA_ORDINALS: [&str; 6] = [
    "eky'okumu",
    "ekyasooka",
    "eky'okubiri",
    "eky'okusatu",
    "eky'okuna",
    "eky'okutaano",
];

static ORDINAL_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b([A-Z][a-zA-Z]+)\s+(I|II|III|IV|V|VI|VII|VIII|IX|X)\b").unwrap()
});

fn roman_to_int(roman: &str) -> Option<usize> {
    match roman {
        "I" => Some(1),
        "II" => Some(2),
        "III" => Some(3),
        "IV" => Some(4),
        "V" => Some(5),
        "VI" => Some(6),
        "VII" => Some(7),
        "VIII" => Some(8),
        "IX" => Some(9),
        "X" => Some(10),
        _ => None,
    }
}

pub fn convert_ordinal_to_luganda(roman_or_num: &str, context_noun: &str) -> String {
    let token = roman_or_num.trim().to_uppercase();
    let val = roman_to_int(&token).or_else(|| token.parse::<usize>().ok());

    if let Some(v) = val {
        let noun_lower = context_noun.to_lowercase();
        if noun_lower == "ekyasa" || noun_lower == "ekitundu" {
            if v < EKYASA_ORDINALS.len() {
                return EKYASA_ORDINALS[v].to_string();
            }
            return "eky'okumu".to_string();
        }

        if v < ORDINALS_MAP.len() {
            return ORDINALS_MAP[v].to_string();
        }
        return "ow'okumu".to_string();
    }

    roman_or_num.to_string()
}

pub fn replace_ordinals_in_text(text: &str) -> String {
    ORDINAL_RE.replace_all(text, |caps: &regex::Captures| {
        let noun = &caps[1];
        let roman = &caps[2];
        let spoken_ord = convert_ordinal_to_luganda(roman, noun);
        format!("{} {}", noun, spoken_ord)
    }).into_owned()
}
