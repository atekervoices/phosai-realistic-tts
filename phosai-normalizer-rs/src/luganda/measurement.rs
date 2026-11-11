//! Luganda Measurement Verbaliser with Luganda Unit Ordering.

use regex::Regex;
use once_cell::sync::Lazy;
use super::cardinals::convert_cardinal_to_luganda;

static QUOTE_MEASURE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(\d+(?:\.\d+)?)\s*(''|')").unwrap()
});

static UNIT_MEASURE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(\d+(?:\.\d+)?)\s*(kg|mg|g|km|cm|mm|m|ml|cc|ft|miles|inch|l)\b").unwrap()
});

fn map_unit(unit_raw: &str) -> &'static str {
    match unit_raw.to_lowercase().as_str() {
        "kg" => "kkirogulaamu",
        "g" => "gulaamu",
        "mg" => "milligulaamu",
        "km" => "kirommita",
        "m" => "mmita",
        "cm" => "ssentimmita",
        "mm" => "millimmita",
        "l" => "liita",
        "ml" => "milliliita",
        "cc" => "ssisi",
        "ft" => "futi",
        "inch" | "inches" => "yiinci",
        "miles" => "mayiro",
        _ => "mmita",
    }
}

pub fn replace_measurements_in_text(text: &str) -> String {
    // 1. Feet and inches quote symbols: 5' or 5''
    let res = QUOTE_MEASURE_RE.replace_all(text, |caps: &regex::Captures| {
        let val_str = &caps[1];
        let sym = &caps[2];
        let unit = if sym == "'" { "futi" } else { "yiinci" };
        let num_words = convert_cardinal_to_luganda(val_str, 0);
        format!("{} {}", unit, num_words)
    });

    // 2. Standard numeric unit expressions: 10 kg, 24 m, 3 km
    let res = UNIT_MEASURE_RE.replace_all(&res, |caps: &regex::Captures| {
        let val_str = &caps[1];
        let unit_raw = &caps[2];
        let unit_name = map_unit(unit_raw);
        let num_words = convert_cardinal_to_luganda(val_str, 0);
        format!("{} {}", unit_name, num_words)
    });

    res.into_owned()
}
