//! Luganda Date and East African 12-Hour Time Verbaliser.

use regex::Regex;
use once_cell::sync::Lazy;
use super::cardinals::convert_cardinal_integer;

const LUGANDA_MONTHS_ORDINAL: [&str; 13] = [
    "",
    "ogusooka",
    "ogwokubiri",
    "ogwokusatu",
    "ogwokuna",
    "ogwokutaano",
    "ogwomukaaga",
    "ogwomusanvu",
    "ogwomunaana",
    "ogwomwenda",
    "ogwekkumi",
    "ogwekkuminogumu",
    "ogwekkuminebiri",
];

static TIME_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(?:\bssaawa\s+)?\b(\d{1,2}):(\d{2})(?::\d{2})?\s*(am|pm|ez'enkya|ez'ekiro|olw'eggulo)?\b").unwrap()
});

static DATE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(?:\b(?:enaku\s+z'omwezi|olunaku\s+lwa|olunaku|nga)\s+)?\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})\b").unwrap()
});

/// Converts standard 24-hour/12-hour clock time into Luganda East African 12-hour time.
pub fn convert_time_to_luganda(mut hour: u32, minute: u32, period: &str) -> String {
    let period_clean = period.to_lowercase();
    if period_clean.contains("pm") && hour < 12 {
        hour += 12;
    } else if period_clean.contains("am") && hour == 12 {
        hour = 0;
    }

    let mut lug_hour = (hour as i32 - 6).rem_euclid(12) as u32;
    if lug_hour == 0 {
        lug_hour = 12;
    }

    let time_period = if (7..12).contains(&hour) {
        "ez'enkya"
    } else if (12..17).contains(&hour) {
        "olw'eggulo"
    } else if (17..19).contains(&hour) {
        "ez'akawungeezi"
    } else if (19..=23).contains(&hour) || hour < 5 {
        "ez'ekiro"
    } else {
        "ez'obudde"
    };

    let hour_words = convert_cardinal_integer(lug_hour as u64, 0);

    if minute == 0 {
        format!("ssaawa {} {}", hour_words, time_period).trim().to_string()
    } else {
        let min_words = convert_cardinal_integer(minute as u64, 0);
        format!("eddakiika {} eziyise ku ssaawa {} {}", min_words, hour_words, time_period).trim().to_string()
    }
}

pub fn replace_times_in_text(text: &str) -> String {
    TIME_RE.replace_all(text, |caps: &regex::Captures| {
        let h: u32 = caps[1].parse().unwrap_or(0);
        let m: u32 = caps[2].parse().unwrap_or(0);
        let period = caps.get(3).map_or("", |m| m.as_str());
        convert_time_to_luganda(h, m, period)
    }).into_owned()
}

pub fn convert_date_to_luganda(day: u32, month: u32, year: u32) -> String {
    let day_words = convert_cardinal_integer(day as u64, 0);
    let month_words = if (1..=12).contains(&month) {
        LUGANDA_MONTHS_ORDINAL[month as usize]
    } else {
        "ogwa"
    };
    let year_words = convert_cardinal_integer(year as u64, 0).replace("amakumi abiri", "abiri");

    format!("enaku z'omwezi {} {} omwaka {}", day_words, month_words, year_words)
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
            convert_date_to_luganda(d, m, y)
        } else {
            caps[0].to_string()
        }
    });

    YEAR_RANGE_RE.replace_all(&text, |caps: &regex::Captures| {
        let y1_str = &caps[1];
        let y2_str = &caps[2];
        let y1_num: u64 = y1_str.parse().unwrap_or(0);
        let y2_num: u64 = if y2_str.len() == 2 {
            (y1_num / 100) * 100 + y2_str.parse::<u64>().unwrap_or(0)
        } else {
            y2_str.parse::<u64>().unwrap_or(0)
        };
        let y1_words = convert_cardinal_integer(y1_num, 0);
        let y2_words = convert_cardinal_integer(y2_num, 0);
        format!("{} okutuuka ku {}", y1_words, y2_words)
    }).into_owned()
}
