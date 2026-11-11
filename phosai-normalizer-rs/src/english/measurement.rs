//! English Measurement Verbaliser.

use regex::Regex;
use once_cell::sync::Lazy;
use super::numbers::convert_number_to_english;

static UNIT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(\d+(?:\.\d+)?)\s*(kg|mg|g|km|cm|mm|m|ml|ft|in|inch|inches|mi|miles|l)\b").unwrap()
});

fn map_unit_en(unit_raw: &str, is_singular: bool) -> &'static str {
    match unit_raw.to_lowercase().as_str() {
        "kg" => if is_singular { "kilogram" } else { "kilograms" },
        "g" => if is_singular { "gram" } else { "grams" },
        "mg" => if is_singular { "milligram" } else { "milligrams" },
        "km" => if is_singular { "kilometer" } else { "kilometers" },
        "m" => if is_singular { "meter" } else { "meters" },
        "cm" => if is_singular { "centimeter" } else { "centimeters" },
        "mm" => if is_singular { "millimeter" } else { "millimeters" },
        "l" => if is_singular { "liter" } else { "liters" },
        "ml" => if is_singular { "milliliter" } else { "milliliters" },
        "ft" => if is_singular { "foot" } else { "feet" },
        "in" | "inch" | "inches" => if is_singular { "inch" } else { "inches" },
        "mi" | "miles" => if is_singular { "mile" } else { "miles" },
        _ => "units",
    }
}

pub fn replace_measurements_en(text: &str) -> String {
    UNIT_RE.replace_all(text, |caps: &regex::Captures| {
        let val_str = &caps[1];
        let unit_raw = &caps[2];
        let val: f64 = val_str.parse().unwrap_or(0.0);
        let is_singular = (val - 1.0).abs() < f64::EPSILON;
        let unit_word = map_unit_en(unit_raw, is_singular);
        let num_word = convert_number_to_english(val_str);
        format!("{} {}", num_word, unit_word)
    }).into_owned()
}
