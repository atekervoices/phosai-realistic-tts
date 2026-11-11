//! Kinyarwanda Symbol, Percentage, and Fraction Verbaliser.

use regex::Regex;
use once_cell::sync::Lazy;
use super::cardinals::{convert_cardinal_to_kinyarwanda, convert_cardinal_integer};

static PCT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\d+(?:\.\d+)?)\s*%").unwrap());
static HALF_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b1/2\b").unwrap());
static QUARTER_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b1/4\b").unwrap());
static FRAC_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b(\d+)/(\d+)\b").unwrap());

static MATH_PLUS_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\d+)\s*\+\s*(\d+)").unwrap());
static MATH_MINUS_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\d+)\s*-\s*(\d+)").unwrap());
static MATH_MUL_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\d+)\s*\*\s*(\d+)").unwrap());
static MATH_EQ_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\d+)\s*=\s*(\d+)").unwrap());

static AMP_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s*&\s*").unwrap());
static AT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"@").unwrap());
static EQ_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s*=\s*").unwrap());

pub fn replace_symbols_and_percentages_kin(text: &str) -> String {
    // 1. Percentages
    let res = PCT_RE.replace_all(text, |caps: &regex::Captures| {
        let val_words = convert_cardinal_to_kinyarwanda(&caps[1]);
        format!("{} ku ijana", val_words)
    });

    // 2. Fractions
    let res = HALF_RE.replace_all(&res, "igice");
    let res = QUARTER_RE.replace_all(&res, "robo");

    let res = FRAC_RE.replace_all(&res, |caps: &regex::Captures| {
        let num1: i128 = caps[1].parse().unwrap_or(0);
        let num2: i128 = caps[2].parse().unwrap_or(0);
        let w1 = convert_cardinal_integer(num1);
        let w2 = convert_cardinal_integer(num2);
        format!("{} ku {}", w1, w2)
    });

    // 3. Arithmetic operations
    let res = MATH_PLUS_RE.replace_all(&res, "$1 guterana $2");
    let res = MATH_MINUS_RE.replace_all(&res, "$1 gukuramo $2");
    let res = MATH_MUL_RE.replace_all(&res, "$1 gukuba $2");
    let res = MATH_EQ_RE.replace_all(&res, "$1 bingana na $2");

    // 4. Standalone symbols
    let res = AMP_RE.replace_all(&res, " na ");
    let res = AT_RE.replace_all(&res, " at ");
    let res = EQ_RE.replace_all(&res, " bingana na ");

    res.into_owned()
}
