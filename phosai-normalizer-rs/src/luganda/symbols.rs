//! Luganda Symbols, Percentages, and Fractions Verbaliser.

use regex::Regex;
use once_cell::sync::Lazy;
use super::cardinals::convert_cardinal_to_luganda;

static COVID_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)\bCOVID-19\b").unwrap());
static PERCENT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\d+(?:\.\d+)?)\s*%").unwrap());
static HALF_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b1/2\b").unwrap());
static QUARTER_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b1/4\b").unwrap());
static THREE_QUARTERS_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b3/4\b").unwrap());
static AMP_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s*&\s*").unwrap());
static PLUS_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s*\+\s*").unwrap());
static EQUALS_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s*=\s*").unwrap());
static AT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"@").unwrap());

pub fn replace_symbols_and_percentages_lug(text: &str) -> String {
    let res = COVID_RE.replace_all(text, "Kovidi kkumi na mwenda");

    let res = PERCENT_RE.replace_all(&res, |caps: &regex::Captures| {
        let val_str = &caps[1];
        let words = convert_cardinal_to_luganda(val_str, 0);
        format!("{} ku kikumi", words)
    });

    let res = HALF_RE.replace_all(&res, "ekitundu");
    let res = QUARTER_RE.replace_all(&res, "ekitundu ky'okuna");
    let res = THREE_QUARTERS_RE.replace_all(&res, "ebitundu bisatu ku bina");

    let res = AMP_RE.replace_all(&res, " na ");
    let res = PLUS_RE.replace_all(&res, " gatta ");
    let res = EQUALS_RE.replace_all(&res, " kyenkana ");
    let res = AT_RE.replace_all(&res, " at ");

    res.into_owned()
}
