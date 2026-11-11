//! English Cardinal, Ordinal, and Decimal Verbaliser.

use regex::Regex;
use once_cell::sync::Lazy;

const ONES: [&str; 20] = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen"
];

const TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"
];

const RANKS_EN: [(u128, &str); 12] = [
    (10u128.pow(33), "decillion"),
    (10u128.pow(30), "nonillion"),
    (10u128.pow(27), "octillion"),
    (10u128.pow(24), "septillion"),
    (10u128.pow(21), "sextillion"),
    (10u128.pow(18), "quintillion"),
    (10u128.pow(15), "quadrillion"),
    (10u128.pow(12), "trillion"),
    (10u128.pow(9), "billion"),
    (10u128.pow(6), "million"),
    (10u128.pow(3), "thousand"),
    (100, "hundred"),
];

static ORD_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(\d+)(st|nd|rd|th)\b").unwrap()
});

pub fn convert_cardinal_integer_en(num: u128) -> String {
    if num < 20 {
        return ONES[num as usize].to_string();
    }

    if num < 100 {
        let t = (num / 10) as usize;
        let r = (num % 10) as usize;
        if r != 0 {
            return format!("{} {}", TENS[t], ONES[r]);
        }
        return TENS[t].to_string();
    }

    if num < 1000 {
        let h = (num / 100) as usize;
        let r = num % 100;
        let h_str = format!("{} hundred", ONES[h]);
        if r != 0 {
            return format!("{} {}", h_str, convert_cardinal_integer_en(r));
        }
        return h_str;
    }

    if num < 1_000_000 {
        let th = num / 1000;
        let r = num % 1000;
        let th_str = format!("{} thousand", convert_cardinal_integer_en(th));
        if r != 0 {
            return format!("{} {}", th_str, convert_cardinal_integer_en(r));
        }
        return th_str;
    }

    for (val, name) in RANKS_EN {
        if val <= 1000 {
            continue;
        }
        if num >= val {
            let count = num / val;
            let r = num % val;
            let m_str = format!("{} {}", convert_cardinal_integer_en(count), name);
            if r != 0 {
                return format!("{} {}", m_str, convert_cardinal_integer_en(r));
            }
            return m_str;
        }
    }

    num.to_string()
}

pub fn convert_ordinal_en(num: u128) -> String {
    let card_str = convert_cardinal_integer_en(num);
    let mut words: Vec<&str> = card_str.split_whitespace().collect();
    if let Some(last_word) = words.pop() {
        let ordinal_last = match last_word {
            "one" => "first",
            "two" => "second",
            "three" => "third",
            "four" => "fourth",
            "five" => "fifth",
            "six" => "sixth",
            "seven" => "seventh",
            "eight" => "eighth",
            "nine" => "ninth",
            "ten" => "tenth",
            "eleven" => "eleventh",
            "twelve" => "twelfth",
            "twenty" => "twentieth",
            "thirty" => "thirtieth",
            "forty" => "fortieth",
            "fifty" => "fiftieth",
            "sixty" => "sixtieth",
            "seventy" => "seventieth",
            "eighty" => "eightieth",
            "ninety" => "ninetieth",
            _ => {
                if last_word.ends_with('y') {
                    // e.g. twenty -> twentieth handled above
                    "th"
                } else if last_word.ends_with('e') {
                    // e.g. nine -> ninth handled above
                    "th"
                } else {
                    "th"
                }
            }
        };

        if !words.is_empty() {
            if ordinal_last == "th" {
                format!("{} {}th", words.join(" "), last_word)
            } else {
                format!("{} {}", words.join(" "), ordinal_last)
            }
        } else if ordinal_last == "th" {
            format!("{}th", last_word)
        } else {
            ordinal_last.to_string()
        }
    } else {
        "first".to_string()
    }
}

pub fn convert_number_to_english(num_str: &str) -> String {
    let clean_str = num_str.replace(',', "").trim().to_string();

    if let Some((int_str, frac_str)) = clean_str.split_once('.') {
        let int_part: u128 = int_str.parse().unwrap_or(0);
        let int_words = convert_cardinal_integer_en(int_part);
        let frac_words: Vec<String> = frac_str
            .chars()
            .filter_map(|c| c.to_digit(10))
            .map(|d| ONES[d as usize].to_string())
            .collect();
        return format!("{} point {}", int_words, frac_words.join(" "));
    }

    if let Ok(val) = clean_str.parse::<u128>() {
        convert_cardinal_integer_en(val)
    } else {
        num_str.to_string()
    }
}

pub fn replace_ordinals_en(text: &str) -> String {
    ORD_RE.replace_all(text, |caps: &regex::Captures| {
        let val: u128 = caps[1].parse().unwrap_or(1);
        convert_ordinal_en(val)
    }).into_owned()
}
