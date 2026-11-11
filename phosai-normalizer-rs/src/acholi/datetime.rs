//! Acholi Date and East African 12-Hour Time Verbaliser.

use regex::Regex;
use once_cell::sync::Lazy;
use super::cardinals::convert_cardinal_integer;

const ACHOLI_MONTHS_ORDINAL: [&str; 13] = [
    "",
    "Dwe me acel",
    "Dwe me aryo",
    "Dwe me adek",
    "Dwe me angwen",
    "Dwe me abic",
    "Dwe me abicel",
    "Dwe me abiro",
    "Dwe me aboro",
    "Dwe me abungwen",
    "Dwe me apar",
    "Dwe me apar acel",
    "Dwe me apar aryo",
];

static TIME_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(?:\bcawa\s+)?\b(\d{1,2}):(\d{2})(?::\d{2})?\s*(am|pm)?\b").unwrap()
});

static DATE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(?:\bnino\s+)?\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b").unwrap()
});

pub fn convert_time_to_acholi(mut hour: u32, minute: u32, period: &str) -> String {
    let period_clean = period.to_lowercase();
    if period_clean.contains("pm") && hour < 12 {
        hour += 12;
    } else if period_clean.contains("am") && hour == 12 {
        hour = 0;
    }

    let mut ach_hour = (hour as i32 - 6).rem_euclid(12) as u32;
    if ach_hour == 0 {
        ach_hour = 12;
    }

    let time_period = if (5..12).contains(&hour) {
        "me odiko"
    } else if (12..16).contains(&hour) {
        "me dyeceng"
    } else if (16..19).contains(&hour) {
        "me otyeno"
    } else {
        "me dyewor"
    };

    let hour_words = convert_cardinal_integer(ach_hour as i128);

    if minute == 0 {
        return format!("cawa {} {}", hour_words, time_period).trim().to_string();
    }

    let min_words = convert_cardinal_integer(minute as i128);
    format!("cawa {} ki dakika {} {}", hour_words, min_words, time_period).trim().to_string()
}

pub fn replace_times_in_text(text: &str) -> String {
    TIME_RE.replace_all(text, |caps: &regex::Captures| {
        let h: u32 = caps[1].parse().unwrap_or(0);
        let m: u32 = caps[2].parse().unwrap_or(0);
        let period = caps.get(3).map_or("", |m| m.as_str());
        convert_time_to_acholi(h, m, period)
    }).into_owned()
}

pub fn convert_date_to_acholi(day: u32, month: u32, year: u32) -> String {
    let day_words = convert_cardinal_integer(day as i128);
    let month_words = if (1..=12).contains(&month) {
        ACHOLI_MONTHS_ORDINAL[month as usize]
    } else {
        "Dwe me"
    };
    let year_words = convert_cardinal_integer(year as i128);

    format!("nino {} {} mwaka {}", day_words, month_words, year_words)
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
            convert_date_to_acholi(d, m, y)
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
        format!("{} nio wa {}", y1_words, y2_words)
    }).into_owned()
}
