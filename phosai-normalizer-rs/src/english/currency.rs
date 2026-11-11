//! English Currency Verbaliser.

use regex::Regex;
use once_cell::sync::Lazy;
use super::numbers::convert_cardinal_integer_en;

static USD_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\$(\d+(?:[.,]\d+)*)").unwrap()
});

static GBP_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"£(\d+(?:[.,]\d+)*)").unwrap()
});

static UGX_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\bUGX\s*(\d+(?:[.,]\d+)*)\b").unwrap()
});

pub fn replace_currencies_en(text: &str) -> String {
    // 1. USD: $50.25 or $50
    let res = USD_RE.replace_all(text, |caps: &regex::Captures| {
        let amt_str = caps[1].replace(',', "");
        if let Some((m_str, c_str)) = amt_str.split_once('.') {
            let m: u128 = m_str.parse().unwrap_or(0);
            let c: u128 = c_str.parse().unwrap_or(0);
            let m_words = convert_cardinal_integer_en(m);
            let m_unit = if m == 1 { "dollar" } else { "dollars" };
            let c_words = convert_cardinal_integer_en(c);
            let c_unit = if c == 1 { "cent" } else { "cents" };
            format!("{} {} and {} {}", m_words, m_unit, c_words, c_unit)
        } else {
            let m: u128 = amt_str.parse().unwrap_or(0);
            let m_words = convert_cardinal_integer_en(m);
            let m_unit = if m == 1 { "dollar" } else { "dollars" };
            format!("{} {}", m_words, m_unit)
        }
    });

    // 2. GBP: £50
    let res = GBP_RE.replace_all(&res, |caps: &regex::Captures| {
        let amt_str = caps[1].replace(',', "");
        let m: u128 = amt_str.parse().unwrap_or(0);
        let m_words = convert_cardinal_integer_en(m);
        let m_unit = if m == 1 { "pound" } else { "pounds" };
        format!("{} {}", m_words, m_unit)
    });

    // 3. UGX: UGX 50,000
    let res = UGX_RE.replace_all(&res, |caps: &regex::Captures| {
        let amt_str = caps[1].replace(',', "");
        let m: u128 = amt_str.parse().unwrap_or(0);
        let m_words = convert_cardinal_integer_en(m);
        format!("{} Uganda shillings", m_words)
    });

    res.into_owned()
}
