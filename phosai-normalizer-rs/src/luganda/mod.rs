//! Luganda Normalizer Module.

pub mod cardinals;
pub mod datetime;
pub mod currency;
pub mod measurement;
pub mod ordinals;
pub mod phonenumber;
pub mod ranges;
pub mod symbols;
pub mod abbreviations;
pub mod spelling;
pub mod contractions;
pub mod diacritics;

use regex::Regex;
use once_cell::sync::Lazy;
use crate::cleaner::{normalize_unicode, standardize_punctuation, normalize_math_symbols, clean_emojis, clean_whitespace};

static STANDALONE_NUM_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^-?\d+(?:,\d+)*(?:\.\d+)?$").unwrap()
});
static WORD_CLEAN_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"[^\w]").unwrap()
});

#[derive(Default, Debug, Clone)]
pub struct LugandaNormalizer {
    pub expand_elisions: bool,
    pub standardize_typos: bool,
}

impl LugandaNormalizer {
    pub fn new() -> Self {
        Self {
            expand_elisions: false,
            standardize_typos: true,
        }
    }

    pub fn normalize(&self, text: &str) -> String {
        if text.trim().is_empty() {
            return String::new();
        }

        // Step 1: Unicode & Diacritic cleanup, Math normalization
        let text = normalize_unicode(text);
        let text = diacritics::normalize_diacritics(&text);
        let text = standardize_punctuation(&text);
        let text = normalize_math_symbols(&text);
        let text = clean_emojis(&text);

        // Step 2: Spelling Standardization
        let text = if self.standardize_typos {
            spelling::standardize_spelling(&text)
        } else {
            text
        };

        // Step 3: Optional Elision Expansion (n'ekitabo -> na ekitabo)
        let text = if self.expand_elisions {
            contractions::expand_elisions(text)
        } else {
            text
        };

        // Step 4: Phone Numbers (e.g. 070278319)
        let text = phonenumber::replace_phone_numbers_in_text(&text);

        // Step 5: Ordinals (e.g. Muteesa I, Muteesa II)
        let text = ordinals::replace_ordinals_in_text(&text);

        // Step 6: Dates & EA 12-Hour Clock Times (e.g. 16/07/2026, 07:00)
        let text = datetime::replace_dates_in_text(&text);
        let text = datetime::replace_times_in_text(&text);

        // Step 7: Currencies (e.g. UGX 50,000, 50,000/=, $50, $2.50, Ksh 7.38)
        let text = currency::replace_currencies_in_text(&text);

        // Step 8: Measurements (e.g. 24 m, 10 kg, 3 kg, 5')
        let text = measurement::replace_measurements_in_text(&text);

        // Step 9: Ranges (e.g. 10-20)
        let text = ranges::replace_ranges_in_text(&text);

        // Step 10: Symbols, Percentages & Fractions (e.g. 50%, 1/2, &, @, =)
        let text = symbols::replace_symbols_and_percentages_lug(&text);

        // Step 11: Abbreviations & Acronyms (e.g. Mw., Owek., URA, NSSF, UNRA)
        let text = abbreviations::replace_abbreviations_in_text(text);

        // Step 12: Standalone Cardinal Numbers with surrounding Noun-Class Agreement
        let text = self.replace_standalone_cardinals(&text);

        // Step 13: Clean final whitespace
        clean_whitespace(&text)
    }

    fn replace_standalone_cardinals(&self, text: &str) -> String {
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut tokens: Vec<String> = words.into_iter().map(|w| w.to_string()).collect();

        for idx in 0..tokens.len() {
            let token = &tokens[idx];
            let trimmed = token.trim_matches(|c: char| c.is_ascii_punctuation());
            if STANDALONE_NUM_RE.is_match(trimmed) {
                let prev_word = if idx > 0 { &tokens[idx - 1] } else { "" };
                let prev_word_clean = WORD_CLEAN_RE.replace_all(prev_word, "");
                let class_idx = cardinals::detect_noun_class_index(&prev_word_clean);

                let spoken_num = cardinals::convert_cardinal_to_luganda(trimmed, class_idx);
                tokens[idx] = tokens[idx].replace(trimmed, &spoken_num);
            }
        }

        tokens.join(" ")
    }
}
