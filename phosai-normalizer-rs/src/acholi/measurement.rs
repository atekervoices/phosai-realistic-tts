//! Acholi Measurement Verbaliser.

use regex::Regex;
use once_cell::sync::Lazy;
use super::cardinals::convert_cardinal_to_acholi;

static SPEED_KMH_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(\d+(?:\.\d+)?)\s*km/h\b").unwrap()
});
static SPEED_MS_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(\d+(?:\.\d+)?)\s*m/s\b").unwrap()
});
static UNIT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(\d+(?:\.\d+)?)\s*(kg|mg|g|km|cm|mm|m|ml|hrs?|h|mins?|min|sec|s|l)\.?\b").unwrap()
});

fn map_unit(unit_raw: &str) -> &'static str {
    match unit_raw.to_lowercase().as_str() {
        "kg" => "kilogram",
        "g" => "gram",
        "mg" => "milligram",
        "km" => "kilometre",
        "m" => "metre",
        "cm" => "centimetre",
        "mm" => "millimetre",
        "l" => "litir",
        "ml" => "millilitir",
        "h" | "hr" | "hrs" => "cawa",
        "min" | "mins" => "dakika",
        "s" | "sec" => "sekond",
        _ => "metre",
    }
}

pub fn replace_measurements_in_text(text: &str) -> String {
    let res = SPEED_KMH_RE.replace_all(text, |caps: &regex::Captures| {
        let val_words = convert_cardinal_to_acholi(&caps[1]);
        format!("kilometre {} i cawa acel", val_words)
    });

    let res = SPEED_MS_RE.replace_all(&res, |caps: &regex::Captures| {
        let val_words = convert_cardinal_to_acholi(&caps[1]);
        format!("metre {} i sekond acel", val_words)
    });

    let res = UNIT_RE.replace_all(&res, |caps: &regex::Captures| {
        let val_str = &caps[1];
        let unit_raw = &caps[2];
        let ach_unit = map_unit(unit_raw);
        let val_words = convert_cardinal_to_acholi(val_str);
        format!("{} {}", ach_unit, val_words)
    });

    res.into_owned()
}
