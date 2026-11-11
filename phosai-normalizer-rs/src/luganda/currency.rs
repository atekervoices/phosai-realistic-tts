//! Luganda Currency Verbaliser.

use regex::Regex;
use once_cell::sync::Lazy;
use super::cardinals::{convert_cardinal_integer, convert_cardinal_to_luganda};

static SLASH_CURRENCY_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(\d+(?:[.,]\d+)*)/(?:=|-)(?:\s*(?:shillings?|shillingi|ssiringi|sh))?").unwrap()
});

static USD_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\$(\d+(?:[.,]\d+)*)").unwrap()
});

static GBP_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"£(\d+(?:[.,]\d+)*)").unwrap()
});

static UGX_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(?:UGX|Ugx|USh)\s*(\d+(?:[.,]\d+)*)\b").unwrap()
});

static KSH_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\bKsh\s*(\d+(?:[.,]\d+)*)\b").unwrap()
});

pub fn replace_currencies_in_text(text: &str) -> String {
    // 1. Slash currency notation: 50,000/= or 100.50/= or 50/-
    let res = SLASH_CURRENCY_RE.replace_all(text, |caps: &regex::Captures| {
        let amount_str = caps[1].replace(',', "");
        if let Some((main_str, cents_str)) = amount_str.split_once('.') {
            let main_amt: u64 = main_str.parse().unwrap_or(0);
            let cents_amt: u64 = cents_str.parse().unwrap_or(0);
            let main_words = convert_cardinal_integer(main_amt, 0);
            let cents_words = convert_cardinal_integer(cents_amt, 0);
            format!("Ssiringi {} n'ennusu {}", main_words, cents_words)
        } else {
            let main_amt: u64 = amount_str.parse().unwrap_or(0);
            let main_words = convert_cardinal_integer(main_amt, 0);
            format!("Ssiringi {}", main_words)
        }
    });

    // 2. USD currency: $50 or $2.50
    let res = USD_RE.replace_all(&res, |caps: &regex::Captures| {
        let amount_str = caps[1].replace(',', "");
        if let Some((main_str, cents_str)) = amount_str.split_once('.') {
            let main_amt: u64 = main_str.parse().unwrap_or(0);
            let cents_amt: u64 = cents_str.parse().unwrap_or(0);
            let main_words = convert_cardinal_integer(main_amt, 0);
            let cents_words = convert_cardinal_integer(cents_amt, 0);
            format!("Ddoola z'Amerika {} n'ennusu {}", main_words, cents_words)
        } else {
            let main_amt: u64 = amount_str.parse().unwrap_or(0);
            let main_words = convert_cardinal_integer(main_amt, 0);
            format!("Ddoola z'Amerika {}", main_words)
        }
    });

    // 3. GBP currency: £50
    let res = GBP_RE.replace_all(&res, |caps: &regex::Captures| {
        let amount_str = caps[1].replace(',', "");
        let val: u64 = amount_str.parse().unwrap_or(0);
        let words = convert_cardinal_integer(val, 0);
        format!("Pawundi {}", words)
    });

    // 4. UGX prefix: UGX 50,000 or Ugx 34,000
    let res = UGX_RE.replace_all(&res, |caps: &regex::Captures| {
        let amount_str = caps[1].replace(',', "");
        let val: u64 = amount_str.parse().unwrap_or(0);
        let words = convert_cardinal_integer(val, 0);
        format!("Ssiringi za Uganda {}", words)
    });

    // 5. Ksh prefix: Ksh 7.38
    let res = KSH_RE.replace_all(&res, |caps: &regex::Captures| {
        let amount_str = caps[1].replace(',', "");
        let words = convert_cardinal_to_luganda(&amount_str, 0);
        format!("Ssiringi za Kenya {}", words)
    });

    res.into_owned()
}
