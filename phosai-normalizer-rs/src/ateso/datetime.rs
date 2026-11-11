//! Ateso Date and East African 12-Hour Time Verbaliser.

use regex::Regex;
use once_cell::sync::Lazy;
use super::cardinals::convert_cardinal_integer;

const ATESO_MONTHS_NAMED: [&str; 13] = [
    "",
    "Orara",
    "Omuk",
    "Okwang",
    "Odunge",
    "Opedelei",
    "Omaruk",
    "Omodokokingol",
    "Otikoik",
    "Otibar",
    "Osokosokoma",
    "Osuban",
    "Opo",
];

static TIME_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(?:\besaa\s+)?\b(\d{1,2}):(\d{2})(?::\d{2})?\s*(am|pm)?\b").unwrap()
});

static DATE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(?:\b(?:apaaran|namba)\s+)?\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b").unwrap()
});

pub fn convert_time_to_ateso(mut hour: u32, minute: u32, period: &str) -> String {
    let period_clean = period.to_lowercase();
    if period_clean.contains("pm") && hour < 12 {
        hour += 12;
    } else if period_clean.contains("am") && hour == 12 {
        hour = 0;
    }

    let mut ate_hour = (hour as i32 - 6).rem_euclid(12) as u32;
    if ate_hour == 0 {
        ate_hour = 12;
    }

    let time_period = if (5..12).contains(&hour) {
        "taparach"
    } else if (12..16).contains(&hour) {
        "naka ebonis"
    } else if (16..19).contains(&hour) {
        "naka ebwa"
    } else {
        "naka akware"
    };

    let hour_words = convert_cardinal_integer(ate_hour as i128);

    if minute == 0 {
        return format!("esaa {} {}", hour_words, time_period).trim().to_string();
    }

    let min_words = convert_cardinal_integer(minute as i128);
    format!("esaa {} ka idakikai {} {}", hour_words, min_words, time_period).trim().to_string()
}

pub fn replace_times_in_text(text: &str) -> String {
    TIME_RE.replace_all(text, |caps: &regex::Captures| {
        let h: u32 = caps[1].parse().unwrap_or(0);
        let m: u32 = caps[2].parse().unwrap_or(0);
        let period = caps.get(3).map_or("", |m| m.as_str());
        convert_time_to_ateso(h, m, period)
    }).into_owned()
}

pub fn convert_date_to_ateso(day: u32, month: u32, year: u32) -> String {
    let day_words = convert_cardinal_integer(day as i128);
    let month_words = if (1..=12).contains(&month) {
        ATESO_MONTHS_NAMED[month as usize]
    } else {
        "elap"
    };
    let year_words = convert_cardinal_integer(year as i128);

    format!("apar {} {} {}", day_words, month_words, year_words)
}

static YEAR_RANGE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b(\d{4})/(\d{2,4})\b").unwrap()
});

pub fn replace_dates_in_text(text: &str) -> String {
    let text = DATE_RE.replace_all(text, |caps: &regex::Captures| {
        let d: u32 = caps[1].parse().unwrap_or(0);
        let m: u32 = caps[2].parse().unwrap_or(0);
        let mut y: u32 = caps[3].parse().unwrap_or(0);
        if y < 100 {
            y += if y <= 30 { 2000 } else { 1900 };
        }
        if (1..=12).contains(&m) && (1..=31).contains(&d) {
            convert_date_to_ateso(d, m, y)
        } else {
            caps[0].to_string()
        }
    });

    YEAR_RANGE_RE.replace_all(&text, |caps: &regex::Captures| {
        let y1_str = &caps[1];
        let y2_str = &caps[2];
        let y1_num: i128 = y1_str.parse().unwrap_or(0);
        let y2_num: i128 = if y2_str.len() == 2 {
            (y1_num / 100) * 100 + y2_str.parse::<i128>().unwrap_or(0)
        } else {
            y2_str.parse::<i128>().unwrap_or(0)
        };
        let y1_words = convert_cardinal_integer(y1_num);
        let y2_words = convert_cardinal_integer(y2_num);
        format!("{} aitodol {}", y1_words, y2_words)
    }).into_owned()
}
