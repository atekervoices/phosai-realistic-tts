//! Kinyarwanda Measurement Verbaliser.

use regex::Regex;
use once_cell::sync::Lazy;
use super::cardinals::convert_cardinal_to_kinyarwanda;

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
        "kg" => "ibilo",
        "g" => "gramu",
        "mg" => "miligramu",
        "km" => "ibirometero",
        "m" => "metero",
        "cm" => "santimetero",
        "mm" => "milimetero",
        "l" => "litiro",
        "ml" => "mililitiro",
        "ha" => "hegitari",
        "t" => "tani",
        "h" | "hr" | "hrs" => "amasaha",
        "min" | "mins" => "iminota",
        "s" | "sec" => "amasegonda",
        _ => "metero",
    }
}

pub fn replace_measurements_in_text(text: &str) -> String {
    let res = SPEED_KMH_RE.replace_all(text, |caps: &regex::Captures| {
        let val_words = convert_cardinal_to_kinyarwanda(&caps[1]);
        format!("ibirometero {} ku isaha", val_words)
    });

    let res = SPEED_MS_RE.replace_all(&res, |caps: &regex::Captures| {
        let val_words = convert_cardinal_to_kinyarwanda(&caps[1]);
        format!("metero {} ku isegonda", val_words)
    });

    let res = TEMP_RE.replace_all(&res, |caps: &regex::Captures| {
        let val_words = convert_cardinal_to_kinyarwanda(&caps[1]);
        format!("digiri {} za selsiasi", val_words)
    });

    let res = UNIT_RE.replace_all(&res, |caps: &regex::Captures| {
        let val_str = &caps[1];
        let unit_raw = &caps[2];
        let kin_unit = map_unit(unit_raw);
        let val_words = convert_cardinal_to_kinyarwanda(val_str);
        format!("{} {}", kin_unit, val_words)
    });

    res.into_owned()
}
