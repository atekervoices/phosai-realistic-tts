//! Kinyarwanda Normalizer Module.

pub mod cardinals;
pub mod datetime;
pub mod currency;
pub mod measurement;
pub mod ordinals;
pub mod abbreviations;
pub mod symbols;
pub mod phonenumber;

use regex::Regex;
use once_cell::sync::Lazy;
use crate::cleaner::{normalize_unicode, standardize_punctuation, normalize_math_symbols, clean_emojis, clean_whitespace};

static STANDALONE_NUM_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^-?\d+(?:,\d+)*(?:\.\d+)?$").unwrap()
});

#[derive(Default, Debug, Clone)]
pub struct KinyarwandaNormalizer;

impl KinyarwandaNormalizer {
    pub fn new() -> Self {
        Self
    }

    pub fn normalize(&self, text: &str) -> String {
        if text.trim().is_empty() {
            return String::new();
        }

        // Step 1: Unicode, Punctuation, Math, Emojis
        let text = normalize_unicode(text);
        let text = standardize_punctuation(&text);
        let text = normalize_math_symbols(&text);
        let text = clean_emojis(&text);

        // Step 2: Phone Numbers
        let text = phonenumber::replace_phone_numbers_in_text(&text);

        // Step 3: Ordinals (1st, 2nd, wa 1, Kagame II)
        let text = ordinals::replace_ordinals_in_text(&text);

        // Step 4: Dates & Times (16/07/2026, 07:00, 08:30 AM, 14:15)
        let text = datetime::replace_dates_in_text(&text);
        let text = datetime::replace_times_in_text(&text);

        // Step 5: Currencies (RWF 2,000, 2,000 FRw, $50.25, 500/=)
        let text = currency::replace_currencies_in_text(&text);

        // Step 6: Measurements (10 kg, 5 km, 24 m, 3 cm)
        let text = measurement::replace_measurements_in_text(&text);

        // Step 7: Symbols, Percentages & Fractions (50%, 1/2, &, @, =)
        let text = symbols::replace_symbols_and_percentages_kin(&text);

        // Step 8: Abbreviations & Titles (RBC, RISA, RDB, RDF, Dr., Prof., e.g., etc.)
        let text = abbreviations::replace_abbreviations_in_text(text);

        // Step 9: Standalone Cardinal Numbers
        let text = self.replace_standalone_cardinals(&text);

        // Step 10: Clean final whitespace
        clean_whitespace(&text)
    }

    fn replace_standalone_cardinals(&self, text: &str) -> String {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut tokens: Vec<String> = words.into_iter().map(|w| w.to_string()).collect();

        for token in tokens.iter_mut() {
            let trimmed = token.trim_matches(|c: char| c.is_ascii_punctuation());
            if STANDALONE_NUM_RE.is_match(trimmed) {
                let spoken_num = cardinals::convert_cardinal_to_kinyarwanda(trimmed);
                *token = token.replace(trimmed, &spoken_num);
            }
        }

        tokens.join(" ")
    }
}
