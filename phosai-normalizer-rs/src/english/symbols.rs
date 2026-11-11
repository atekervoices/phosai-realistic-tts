//! English Symbols, Percentages, and Fractions Verbaliser.

use regex::Regex;
use once_cell::sync::Lazy;
use super::numbers::convert_number_to_english;

static PCT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\d+(?:\.\d+)?)\s*%").unwrap());
static HALF_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b1/2\b").unwrap());
static THIRD_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b1/3\b").unwrap());
static QUARTER_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b1/4\b").unwrap());
static TWO_THIRDS_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b2/3\b").unwrap());
static THREE_QUARTERS_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b3/4\b").unwrap());

static AMP_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s*&\s*").unwrap());
static PLUS_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s*\+\s*").unwrap());
static EQ_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s*=\s*").unwrap());
static AT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"@").unwrap());

pub fn replace_symbols_and_percentages_en(text: &str) -> String {
    let res = PCT_RE.replace_all(text, |caps: &regex::Captures| {
        let words = convert_number_to_english(&caps[1]);
        format!("{} percent", words)
    });

    let res = HALF_RE.replace_all(&res, "one half");
    let res = THIRD_RE.replace_all(&res, "one third");
    let res = QUARTER_RE.replace_all(&res, "one quarter");
    let res = TWO_THIRDS_RE.replace_all(&res, "two thirds");
    let res = THREE_QUARTERS_RE.replace_all(&res, "three quarters");

    let res = AMP_RE.replace_all(&res, " and ");
    let res = PLUS_RE.replace_all(&res, " plus ");
    let res = EQ_RE.replace_all(&res, " equals ");
    let res = AT_RE.replace_all(&res, " at ");

    res.into_owned()
}
