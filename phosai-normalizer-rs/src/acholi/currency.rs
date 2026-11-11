//! Acholi Currency Verbaliser.

use regex::Regex;
use once_cell::sync::Lazy;
use super::cardinals::convert_cardinal_integer;

static SLASH_CURRENCY_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(\d+(?:[.,]\d+)*)/(?:=|-)(?:\s*(?:shillings?|shillingi|sh))?").unwrap()
});

static UGX_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(?:UGX|Ugx|USh|Ush)\.?\s*(\d+(?:[.,]\d+)*)\b").unwrap()
});

static KES_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(?:KES|Ksh|KSh)\.?\s*(\d+(?:[.,]\d+)*)\b").unwrap()
});

static TZS_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(?:TZS|Tsh|TSh)\.?\s*(\d+(?:[.,]\d+)*)\b").unwrap()
});

static USD_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(?:\$|USD\s*)(\d+(?:[.,]\d+)*)").unwrap()
});

static EUR_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(?:€|EUR\s*)(\d+(?:[.,]\d+)*)").unwrap()
});

static GBP_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(?:£|GBP\s*)(\d+(?:[.,]\d+)*)").unwrap()
});

fn format_amount(amount_str: &str, currency_name: &str) -> String {
    let clean_amt = amount_str.replace(',', "").trim().to_string();
    if let Some((main_str, cents_str)) = clean_amt.split_once('.') {
        let main_amt: i128 = main_str.parse().unwrap_or(0);
        let cents_amt: i128 = cents_str.parse().unwrap_or(0);
        let main_words = convert_cardinal_integer(main_amt);
        let cents_words = convert_cardinal_integer(cents_amt);
        format!("{} {} ki cent {}", currency_name, main_words, cents_words)
    } else {
        let main_amt: i128 = clean_amt.parse().unwrap_or(0);
        let main_words = convert_cardinal_integer(main_amt);
        format!("{} {}", currency_name, main_words)
    }
}

pub fn replace_currencies_in_text(text: &str) -> String {
    let res = SLASH_CURRENCY_RE.replace_all(text, |caps: &regex::Captures| {
        format_amount(&caps[1], "Shilling")
    });

    let res = UGX_RE.replace_all(&res, |caps: &regex::Captures| {
        format_amount(&caps[1], "Shilling me Uganda")
    });

    let res = KES_RE.replace_all(&res, |caps: &regex::Captures| {
        format_amount(&caps[1], "Shilling me Kenya")
    });

    let res = TZS_RE.replace_all(&res, |caps: &regex::Captures| {
        format_amount(&caps[1], "Shilling me Tanzania")
    });

    let res = USD_RE.replace_all(&res, |caps: &regex::Captures| {
        format_amount(&caps[1], "Dola")
    });

    let res = EUR_RE.replace_all(&res, |caps: &regex::Captures| {
        format_amount(&caps[1], "Euro")
    });

    let res = GBP_RE.replace_all(&res, |caps: &regex::Captures| {
        format_amount(&caps[1], "Pound")
    });

    res.into_owned()
}
