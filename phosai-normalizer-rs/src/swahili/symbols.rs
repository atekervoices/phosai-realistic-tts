//! Swahili (Kiswahili) Symbol, Percentage, and Fraction Verbaliser.

use regex::Regex;
use once_cell::sync::Lazy;
use super::cardinals::{convert_cardinal_to_swahili, convert_cardinal_integer};

static PCT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\d+(?:\.\d+)?)\s*%").unwrap());
static HALF_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b1/2\b").unwrap());
static QUARTER_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b1/4\b").unwrap());
static THREE_QUARTERS_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b3/4\b").unwrap());
static FRAC_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b(\d+)/(\d+)\b").unwrap());

static MATH_PLUS_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\d+)\s*\+\s*(\d+)").unwrap());
static MATH_MINUS_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\d+)\s*-\s*(\d+)").unwrap());
static MATH_MUL_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\d+)\s*\*\s*(\d+)").unwrap());
static MATH_EQ_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\d+)\s*=\s*(\d+)").unwrap());

static AMP_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s*&\s*").unwrap());
static AT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"@").unwrap());
static EQ_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s*=\s*").unwrap());

pub fn replace_symbols_and_percentages_swa(text: &str) -> String {
    // 1. Percentages
    let res = PCT_RE.replace_all(text, |caps: &regex::Captures| {
        let val_words = convert_cardinal_to_swahili(&caps[1], true);
        format!("asilimia {}", val_words)
    });

    // 2. Fractions
    let res = HALF_RE.replace_all(&res, "nusu");
    let res = QUARTER_RE.replace_all(&res, "robo");
    let res = THREE_QUARTERS_RE.replace_all(&res, "robo tatu");

    let res = FRAC_RE.replace_all(&res, |caps: &regex::Captures| {
        let num1: i128 = caps[1].parse().unwrap_or(0);
        let num2: i128 = caps[2].parse().unwrap_or(0);
        let w1 = convert_cardinal_integer(num1, true);
        let w2 = convert_cardinal_integer(num2, true);
        format!("{} juu ya {}", w1, w2)
    });

    // 3. Arithmetic operations
    let res = MATH_PLUS_RE.replace_all(&res, "$1 kujumlisha $2");
    let res = MATH_MINUS_RE.replace_all(&res, "$1 kutoa $2");
    let res = MATH_MUL_RE.replace_all(&res, "$1 kuzidisha $2");
    let res = MATH_EQ_RE.replace_all(&res, "$1 sawa na $2");

    // 4. Standalone symbols
    let res = AMP_RE.replace_all(&res, " na ");
    let res = AT_RE.replace_all(&res, " at ");
    let res = EQ_RE.replace_all(&res, " sawa na ");

    res.into_owned()
}
