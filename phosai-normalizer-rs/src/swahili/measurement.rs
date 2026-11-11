//! Swahili (Kiswahili) Measurement Verbaliser.

use regex::Regex;
use once_cell::sync::Lazy;
use super::cardinals::convert_cardinal_to_swahili;

static SPEED_KMH_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(\d+(?:\.\d+)?)\s*km/h\b").unwrap()
});
static SPEED_MS_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(\d+(?:\.\d+)?)\s*m/s\b").unwrap()
});
static TEMP_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(\d+(?:\.\d+)?)\s*(?:°C|deg\s*C)\b").unwrap()
});
static UNIT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(\d+(?:\.\d+)?)\s*(kg|mg|g|km|cm|mm|m|ml|ha|t|hrs?|h|mins?|min|sec|s|l)\.?\b").unwrap()
});

fn map_unit(unit_raw: &str) -> &'static str {
    match unit_raw.to_lowercase().as_str() {
        "kg" => "kilogramu",
        "g" => "gramu",
        "mg" => "miligramu",
        "km" => "kilometri",
        "m" => "mita",
        "cm" => "sentimita",
        "mm" => "milimita",
        "l" => "lita",
        "ml" => "mililita",
        "ha" => "hektari",
        "t" => "tani",
        "h" | "hr" | "hrs" => "saa",
        "min" | "mins" => "dakika",
        "s" | "sec" => "sekunde",
        _ => "mita",
    }
}

pub fn replace_measurements_in_text(text: &str) -> String {
    let res = SPEED_KMH_RE.replace_all(text, |caps: &regex::Captures| {
        let val_words = convert_cardinal_to_swahili(&caps[1], true);
        format!("{} kilometri kwa saa", val_words)
    });

    let res = SPEED_MS_RE.replace_all(&res, |caps: &regex::Captures| {
        let val_words = convert_cardinal_to_swahili(&caps[1], true);
        format!("{} mita kwa sekunde", val_words)
    });

    let res = TEMP_RE.replace_all(&res, |caps: &regex::Captures| {
        let val_words = convert_cardinal_to_swahili(&caps[1], true);
        format!("digrii {} za selsiasi", val_words)
    });

    let res = UNIT_RE.replace_all(&res, |caps: &regex::Captures| {
        let val_str = &caps[1];
        let unit_raw = &caps[2];
        let sw_unit = map_unit(unit_raw);
        let val_words = convert_cardinal_to_swahili(val_str, true);
        format!("{} {}", sw_unit, val_words)
    });

    res.into_owned()
}
