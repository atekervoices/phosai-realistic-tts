//! English Normalizer Module.

pub mod numbers;
pub mod datetime;
pub mod currency;
pub mod measurement;
pub mod abbreviations;
pub mod phone;
pub mod symbols;

use regex::Regex;
use once_cell::sync::Lazy;
use crate::cleaner::{normalize_unicode, standardize_punctuation, normalize_math_symbols, clean_emojis, clean_whitespace};

static STANDALONE_NUM_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^-?\d+(?:,\d+)*(?:\.\d+)?$").unwrap()
});

#[derive(Default, Debug, Clone)]
pub struct EnglishNormalizer;

impl EnglishNormalizer {
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
        let text = phone::replace_phone_numbers_en(&text);

        // Step 3: Ordinals (1st, 2nd, 3rd, 22nd)
        let text = numbers::replace_ordinals_en(&text);

        // Step 4: Dates & Times (12/04/2026, 08:30 AM)
        let text = datetime::replace_dates_en(&text);
        let text = datetime::replace_times_en(&text);

        // Step 5: Currencies ($50.25, £20, UGX 50,000)
        let text = currency::replace_currencies_en(&text);

        // Step 6: Measurements (10 kg, 5 in, 3 km)
        let text = measurement::replace_measurements_en(&text);

        // Step 7: Symbols, Percentages & Fractions (50%, 1/2, &, @, =)
        let text = symbols::replace_symbols_and_percentages_en(&text);

        // Step 8: Abbreviations & Titles (Dr., Mr., e.g., etc.)
        let text = abbreviations::replace_abbreviations_en(text);

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
                let spoken_num = numbers::convert_number_to_english(trimmed);
                *token = token.replace(trimmed, &spoken_num);
            }
        }

        tokens.join(" ")
    }
}
