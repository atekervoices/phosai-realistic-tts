//! English Date and Time Verbaliser.

use regex::Regex;
use once_cell::sync::Lazy;
use super::numbers::{convert_cardinal_integer_en, convert_ordinal_en};

const ENGLISH_MONTHS: [&str; 13] = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
];

static TIME_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(\d{1,2}):(\d{2})(?::\d{2})?\s*(am|pm|a\.m\.|p\.m\.)?\b").unwrap()
});

static DATE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b").unwrap()
});

pub fn convert_time_en(hour: u32, minute: u32, period: &str) -> String {
    let mut period_clean = period.to_uppercase().replace('.', "");
    if period_clean.is_empty() && hour >= 12 {
        period_clean = if hour < 24 { "PM".to_string() } else { "AM".to_string() };
    } else if period_clean.is_empty() && hour < 12 {
        period_clean = "AM".to_string();
    }

    let h_disp = if (1..=12).contains(&hour) {
        hour
    } else {
        let mod_h = hour % 12;
        if mod_h == 0 { 12 } else { mod_h }
    };

    let h_words = convert_cardinal_integer_en(h_disp as u128);

    let time_str = if minute == 0 {
        format!("{} o'clock", h_words)
    } else if minute < 10 {
        format!("{} oh {}", h_words, convert_cardinal_integer_en(minute as u128))
    } else {
        format!("{} {}", h_words, convert_cardinal_integer_en(minute as u128))
    };

    if !period_clean.is_empty() {
        let p_spoken = if period_clean.contains("AM") { "A M" } else { "P M" };
        format!("{} {}", time_str, p_spoken)
    } else {
        time_str
    }
}

pub fn replace_times_en(text: &str) -> String {
    TIME_RE.replace_all(text, |caps: &regex::Captures| {
        let h: u32 = caps[1].parse().unwrap_or(0);
        let m: u32 = caps[2].parse().unwrap_or(0);
        let period = caps.get(3).map_or("", |m| m.as_str());
        convert_time_en(h, m, period)
    }).into_owned()
}

pub fn convert_date_en(month: u32, day: u32, year: u32) -> String {
    let month_name = if (1..=12).contains(&month) {
        ENGLISH_MONTHS[month as usize]
    } else {
        "January"
    };
    let day_word = convert_ordinal_en(day as u128);

    let year_word = if (2000..2010).contains(&year) {
        format!("two thousand {}", convert_cardinal_integer_en((year - 2000) as u128))
    } else if (2010..2100).contains(&year) {
        let y1 = year / 100;
        let y2 = year % 100;
        format!("{} {}", convert_cardinal_integer_en(y1 as u128), convert_cardinal_integer_en(y2 as u128))
    } else {
        convert_cardinal_integer_en(year as u128)
    };

    format!("{} {}, {}", month_name, day_word, year_word)
}

static YEAR_RANGE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(\d{4})/(\d{2,4})\b").unwrap()
});

pub fn replace_dates_en(text: &str) -> String {
    let text = DATE_RE.replace_all(text, |caps: &regex::Captures| {
        let first: u32 = caps[1].parse().unwrap_or(0);
        let second: u32 = caps[2].parse().unwrap_or(0);
        let mut y: u32 = caps[3].parse().unwrap_or(0);
        if y < 100 {
            y += if y <= 30 { 2000 } else { 1900 };
        }

        let (m, d) = if first > 12 && (1..=12).contains(&second) && (1..=31).contains(&first) {
            // Unambiguously DD/MM/YYYY (e.g. 16/07/2026 -> month 7, day 16)
            (second, first)
        } else if second > 12 && (1..=12).contains(&first) && (1..=31).contains(&second) {
            // Unambiguously MM/DD/YYYY (e.g. 07/16/2026 -> month 7, day 16)
            (first, second)
        } else if (1..=31).contains(&first) && (1..=12).contains(&second) {
            // Default DD/MM/YYYY (e.g. 05/08/2026 -> month 8, day 5)
            (second, first)
        } else {
            (first, second)
        };

        if (1..=12).contains(&m) && (1..=31).contains(&d) {
            convert_date_en(m, d, y)
        } else {
            caps[0].to_string()
        }
    });

    YEAR_RANGE_RE.replace_all(&text, |caps: &regex::Captures| {
        let y1_str = &caps[1];
        let y2_str = &caps[2];
        let y1_val: u32 = y1_str.parse().unwrap_or(0);
        let y2_val: u32 = if y2_str.len() == 2 {
            (y1_val / 100) * 100 + y2_str.parse::<u32>().unwrap_or(0)
        } else {
            y2_str.parse::<u32>().unwrap_or(0)
        };
        let y1_words = convert_cardinal_integer_en(y1_val as u128);
        let y2_words = convert_cardinal_integer_en(y2_val as u128);
        format!("{} to {}", y1_words, y2_words)
    }).into_owned()
}
