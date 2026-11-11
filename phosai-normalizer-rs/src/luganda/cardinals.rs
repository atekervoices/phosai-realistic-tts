//! Luganda Cardinal Number Verbalisation with Noun Class Agreement (Classes I..VI).
//! Grounded on Sula Kagumire & Wayne Okello's SAIEE 2020 paper rules.

use regex::Regex;
use once_cell::sync::Lazy;

// Noun Class agreement forms for numbers 1 to 5:
// Index 0: Class III (default / invariant nouns, currencies, units e.g. emu, bbiri, ssatu, nnya, ttaano)
// Index 1: Class I (people e.g. omu, babiri, basatu, bana, bataano)
// Index 2: Class II (trees/rivers e.g. gumu, ebiri, esatu, ena, etaano)
// Index 3: Class IV (things e.g. kimu, bibiri, bisatu, bina, bitaano)
// Index 4: Class V (names/years/fruits e.g. limu, abiri, asatu, ana, ataano)
// Index 5: Class VI (small things/children e.g. kamu, bubiri, busatu, buna, butaano)

const ONES: [&[&str]; 10] = [
    &["zeero"],
    &["emu", "omu", "gumu", "kimu", "limu", "kamu", "kumu", "lumu"],
    &["bbiri", "babiri", "ebiri", "bibiri", "abiri", "bubiri"],
    &["ssatu", "basatu", "esatu", "bisatu", "asatu", "busatu"],
    &["nnya", "bana", "ena", "bina", "ana", "buna"],
    &["ttaano", "bataano", "etaano", "bitaano", "ataano", "butaano"],
    &["mukaaga", "mukaaga", "mukaaga", "mukaaga", "mukaaga", "mukaaga"],
    &["musanvu", "musanvu", "musanvu", "musanvu", "musanvu", "musanvu"],
    &["munaana", "munaana", "munaana", "munaana", "munaana", "munaana"],
    &["mwenda", "mwenda", "mwenda", "mwenda", "mwenda", "mwenda"],
];

const RONES: [&[&str]; 10] = [
    &["kkumi", "kkumi", "kkumi", "kkumi", "kkumi", "kkumi"],
    &["kkumi n'emu", "kkumi n'omu", "kkumi na gumu", "kkumi na kimu", "kkumi na limu", "kkumi na kamu"],
    &["kkumi na bbiri", "kkumi na babiri", "kkumi na ebiri", "kkumi na bibiri", "kkumi na abiri", "kkumi na bubiri"],
    &["kkumi na ssatu", "kkumi na basatu", "kkumi na esatu", "kkumi na bisatu", "kkumi na asatu", "kkumi na busatu"],
    &["kkumi na nnya", "kkumi na bana", "kkumi na ena", "kkumi na bina", "kkumi na ana", "kkumi na buna"],
    &["kkumi na ttaano", "kkumi na bataano", "kkumi na etaano", "kkumi na bitaano", "kkumi na ataano", "kkumi na butaano"],
    &["kkumi na mukaaga", "kkumi na mukaaga", "kkumi na mukaaga", "kkumi na mukaaga", "kkumi na mukaaga", "kkumi na mukaaga"],
    &["kkumi na musanvu", "kkumi na musanvu", "kkumi na musanvu", "kkumi na musanvu", "kkumi na musanvu", "kkumi na musanvu"],
    &["kkumi na munaana", "kkumi na munaana", "kkumi na munaana", "kkumi na munaana", "kkumi na munaana", "kkumi na munaana"],
    &["kkumi na mwenda", "kkumi na mwenda", "kkumi na mwenda", "kkumi na mwenda", "kkumi na mwenda", "kkumi na mwenda"],
];

const TENS: [&str; 10] = ["", "", "abiri", "asatu", "ana", "ataano", "nkaaga", "nsanvu", "kinaana", "kyenda"];
const HUNDREDS: [&str; 10] = ["", "kikumi", "bibiri", "bisatu", "bina", "bitaano", "lukaaga", "lusanvu", "lunaana", "lwenda"];
const THOUSANDS: [&str; 10] = ["", "lukumi", "nkumi bbiri", "nkumi ssatu", "nkumi nnya", "nkumi ttaano", "kakaaga", "kasanvu", "kanaana", "kenda"];
const TEN_THOUSANDS: [&str; 10] = [
    "", "mutwaalo gumu", "emitwaalo ebiri", "emitwaalo esatu", "emitwaalo ena",
    "emitwaalo etaano", "emitwaalo mukaaga", "emitwaalo musanvu", "emitwaalo munaana", "emitwaalo mwenda"
];
const HUNDRED_THOUSANDS: [&str; 10] = [
    "", "akasiriivu kamu", "obusiriivu bubiri", "obusiriivu busatu", "obusiriivu buna",
    "obusiriivu butaano", "obusiriivu mukaaga", "obusiriivu musanvu", "obusiriivu munaana", "obusiriivu mwenda"
];

static CLASS_I_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^(aba|omu|omuntu|abantu|omlenzi|abalenzi|omusajja|abasajja|omukazi|abakazi)").unwrap());
static CLASS_II_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^(emicungwa|emiti|omuti|emyalo|omwalo)").unwrap());
static CLASS_IV_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^(ebi|eki|ebintu|ekintu|ebitabo|ekitabo|ebipipa)").unwrap());
static CLASS_V_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^(emyaka|amannya|amaato|amaaso|emifaaliso)").unwrap());
static CLASS_VI_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^(obu|aka|obwana|akaana|obuntu|akantu)").unwrap());

/// Detects the Noun Class index (0 to 5) of the word preceding a cardinal number.
pub fn detect_noun_class_index(prev_word: &str) -> usize {
    if prev_word.is_empty() {
        return 0;
    }
    let word = prev_word.to_lowercase();
    let word = word.trim();

    if CLASS_I_RE.is_match(word) {
        1
    } else if CLASS_II_RE.is_match(word) {
        2
    } else if CLASS_IV_RE.is_match(word) {
        3
    } else if CLASS_V_RE.is_match(word) {
        4
    } else if CLASS_VI_RE.is_match(word) {
        5
    } else {
        0
    }
}

pub fn get_ones_word(digit: usize, class_idx: usize) -> &'static str {
    if digit >= ONES.len() {
        return "zeero";
    }
    let forms = ONES[digit];
    if class_idx < forms.len() {
        forms[class_idx]
    } else {
        forms[0]
    }
}

fn get_rones_word(digit: usize, class_idx: usize) -> &'static str {
    if digit >= RONES.len() {
        return "kkumi";
    }
    let forms = RONES[digit];
    if class_idx < forms.len() {
        forms[class_idx]
    } else {
        forms[0]
    }
}

/// Converts an unsigned integer into spoken Luganda words matching class_idx.
pub fn convert_cardinal_integer(num: u64, class_idx: usize) -> String {
    if num == 0 {
        return ONES[0][0].to_string();
    }

    if num < 10 {
        return get_ones_word(num as usize, class_idx).to_string();
    }

    if num < 20 {
        return get_rones_word((num - 10) as usize, class_idx).to_string();
    }

    let s = num.to_string();
    let length = s.len();

    if length == 2 {
        let tens_digit = (num / 10) as usize;
        let ones_digit = (num % 10) as usize;
        let tens_word = if [2, 3, 4, 5].contains(&tens_digit) {
            format!("amakumi {}", TENS[tens_digit])
        } else {
            TENS[tens_digit].to_string()
        };

        if ones_digit == 0 {
            return tens_word;
        }
        return format!("{} mu {}", tens_word, get_ones_word(ones_digit, class_idx));
    }

    if length == 3 {
        let h_digit = (num / 100) as usize;
        let rem = num % 100;
        let h_word = HUNDREDS[h_digit];
        if rem == 0 {
            return h_word.to_string();
        }
        return format!("{} mu {}", h_word, convert_cardinal_integer(rem, class_idx));
    }

    if length == 4 {
        let th_digit = (num / 1000) as usize;
        let rem = num % 1000;
        let th_word = THOUSANDS[th_digit];
        if rem == 0 {
            return th_word.to_string();
        }
        return format!("{} mu {}", th_word, convert_cardinal_integer(rem, class_idx));
    }

    if length == 5 {
        let tt_digit = (num / 10000) as usize;
        let rem = num % 10000;
        let tt_word = TEN_THOUSANDS[tt_digit];
        if rem == 0 {
            return tt_word.to_string();
        }
        return format!("{} mu {}", tt_word, convert_cardinal_integer(rem, class_idx));
    }

    if length == 6 {
        let ht_digit = (num / 100000) as usize;
        let rem = num % 100000;
        let ht_word = HUNDRED_THOUSANDS[ht_digit];
        if rem == 0 {
            return ht_word.to_string();
        }
        return format!("{} mu {}", ht_word, convert_cardinal_integer(rem, class_idx));
    }

    if (7..=9).contains(&length) {
        let m_part = num / 1_000_000;
        let rem = num % 1_000_000;
        let m_word = if m_part > 1 {
            format!("obukadde {}", convert_cardinal_integer(m_part, 0))
        } else {
            "kakadde kamu".to_string()
        };
        if rem == 0 {
            return m_word;
        }
        return format!("{} mu {}", m_word, convert_cardinal_integer(rem, class_idx));
    }

    if length >= 10 {
        let b_part = num / 1_000_000_000;
        let rem = num % 1_000_000_000;
        let b_word = if b_part > 1 {
            format!("obuwumbi {}", convert_cardinal_integer(b_part, 0))
        } else {
            "kawumbi kamu".to_string()
        };
        if rem == 0 {
            return b_word;
        }
        return format!("{} mu {}", b_word, convert_cardinal_integer(rem, class_idx));
    }

    num.to_string()
}

/// Verbalizes cardinal string (including decimals) to Luganda words.
pub fn convert_cardinal_to_luganda(num_str: &str, class_idx: usize) -> String {
    let clean_str = num_str.replace(',', "").trim().to_string();

    if let Some((int_str, frac_str)) = clean_str.split_once('.') {
        let int_part: u64 = int_str.parse().unwrap_or(0);
        let int_words = convert_cardinal_integer(int_part, class_idx);
        let frac_words: Vec<String> = frac_str
            .chars()
            .filter_map(|c| c.to_digit(10))
            .map(|d| get_ones_word(d as usize, 0).to_string())
            .collect();
        return format!("{} katonyeze {}", int_words, frac_words.join(" "));
    }

    if let Ok(val) = clean_str.parse::<u64>() {
        convert_cardinal_integer(val, class_idx)
    } else {
        num_str.to_string()
    }
}
