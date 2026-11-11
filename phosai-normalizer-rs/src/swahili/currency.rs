//! Swahili (Kiswahili) Currency Verbaliser.

use regex::Regex;
use once_cell::sync::Lazy;
use super::cardinals::convert_cardinal_integer;

static SLASH_CURRENCY_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(\d+(?:[.,]\d+)*)/(?:=|-)(?:\s*(?:shillings?|shillingi|sh))?").unwrap()
});

static TZS_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(?:TZS|Tsh|TSh)\.?\s*(\d+(?:[.,]\d+)*)\b").unwrap()
});

static KES_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(?:KES|Ksh|KSh)\.?\s*(\d+(?:[.,]\d+)*)\b").unwrap()
});

static UGX_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(?:UGX|Ugx|USh|Ush)\.?\s*(\d+(?:[.,]\d+)*)\b").unwrap()
});

static RWF_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(?:RWF|FRw)\.?\s*(\d+(?:[.,]\d+)*)\b").unwrap()
});

static SH_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\bSh\.?\s*(\d+(?:[.,]\d+)*)\b").unwrap()
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

fn format_amount(amount_str: &str, currency_name: &str, use_lakh: bool) -> String {
    let clean_amt = amount_str.replace(',', "").trim().to_string();
    if let Some((main_str, cents_str)) = clean_amt.split_once('.') {
        let main_amt: i128 = main_str.parse().unwrap_or(0);
        let cents_amt: i128 = cents_str.parse().unwrap_or(0);
        let main_words = convert_cardinal_integer(main_amt, use_lakh);
        let cents_words = convert_cardinal_integer(cents_amt, use_lakh);
        format!("{} {} na senti {}", currency_name, main_words, cents_words)
    } else {
        let main_amt: i128 = clean_amt.parse().unwrap_or(0);
        let main_words = convert_cardinal_integer(main_amt, use_lakh);
        format!("{} {}", currency_name, main_words)
    }
}

pub fn replace_currencies_in_text(text: &str, use_lakh: bool) -> String {
    let res = SLASH_CURRENCY_RE.replace_all(text, |caps: &regex::Captures| {
        format_amount(&caps[1], "shilingi", use_lakh)
    });

    let res = TZS_RE.replace_all(&res, |caps: &regex::Captures| {
        format_amount(&caps[1], "shilingi za tanzania", use_lakh)
    });

    let res = KES_RE.replace_all(&res, |caps: &regex::Captures| {
        format_amount(&caps[1], "shilingi za kenya", use_lakh)
    });

    let res = UGX_RE.replace_all(&res, |caps: &regex::Captures| {
        format_amount(&caps[1], "shilingi za uganda", use_lakh)
    });

    let res = RWF_RE.replace_all(&res, |caps: &regex::Captures| {
        format_amount(&caps[1], "faranga za rwanda", use_lakh)
    });

    let res = SH_RE.replace_all(&res, |caps: &regex::Captures| {
        format_amount(&caps[1], "shilingi", use_lakh)
    });

    let res = USD_RE.replace_all(&res, |caps: &regex::Captures| {
        format_amount(&caps[1], "dola", use_lakh)
    });

    let res = EUR_RE.replace_all(&res, |caps: &regex::Captures| {
        format_amount(&caps[1], "euro", use_lakh)
    });

    let res = GBP_RE.replace_all(&res, |caps: &regex::Captures| {
        format_amount(&caps[1], "pauni", use_lakh)
    });

    res.into_owned()
}
