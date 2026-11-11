//! Swahili (Kiswahili) Cardinal Number Verbaliser.

const UNITS: [&str; 10] = [
    "sifuri", "moja", "mbili", "tatu", "nne", "tano", "sita", "saba", "nane", "tisa"
];

const TENS: [&str; 10] = [
    "", "kumi", "ishirini", "thelathini", "arobaini", "hamsini", "sitini", "sabini", "themanini", "tisini"
];

const RANKS: [(u128, &str); 11] = [
    (10u128.pow(33), "desilioni"),
    (10u128.pow(30), "nonilioni"),
    (10u128.pow(27), "oktilioni"),
    (10u128.pow(24), "septilioni"),
    (10u128.pow(21), "seksitilioni"),
    (10u128.pow(18), "kuintilioni"),
    (10u128.pow(15), "kuadrilioni"),
    (10u128.pow(12), "trilioni"),
    (10u128.pow(9), "bilioni"),
    (10u128.pow(6), "milioni"),
    (10u128.pow(3), "elfu"),
];

pub fn convert_cardinal_integer(n: i128, use_lakh: bool) -> String {
    if n < 0 {
        return format!("hasi {}", convert_cardinal_integer(n.abs(), use_lakh));
    }
    let n = n as u128;

    if n < 10 {
        return UNITS[n as usize].to_string();
    }

    if n < 100 {
        let ten_val = (n / 10) as usize;
        let rem = (n % 10) as usize;
        if rem == 0 {
            return TENS[ten_val].to_string();
        }
        return format!("{} na {}", TENS[ten_val], UNITS[rem]);
    }

    if n < 1000 {
        let hundred_val = (n / 100) as usize;
        let rem = n % 100;
        let h_str = format!("mia {}", UNITS[hundred_val]);
        if rem == 0 {
            return h_str;
        }
        return format!("{} na {}", h_str, convert_cardinal_integer(rem as i128, use_lakh));
    }

    if use_lakh && (100_000..1_000_000).contains(&n) {
        let laki_val = (n / 100_000) as usize;
        let rem = n % 100_000;
        let l_str = format!("laki {}", UNITS[laki_val]);
        if rem == 0 {
            return l_str;
        }
        return format!("{} na {}", l_str, convert_cardinal_integer(rem as i128, use_lakh));
    }

    if n < 1_000_000 {
        let thousand_val = n / 1000;
        let rem = n % 1000;
        let t_str = format!("elfu {}", convert_cardinal_integer(thousand_val as i128, use_lakh));
        if rem == 0 {
            return t_str;
        }
        if rem < 100 {
            return format!("{} na {}", t_str, convert_cardinal_integer(rem as i128, use_lakh));
        }
        return format!("{} {}", t_str, convert_cardinal_integer(rem as i128, use_lakh));
    }

    for (val, name) in RANKS {
        if val == 1000 {
            continue;
        }
        if n >= val {
            let count = n / val;
            let rem = n % val;
            let m_str = format!("{} {}", name, convert_cardinal_integer(count as i128, use_lakh));
            if rem == 0 {
                return m_str;
            }
            if rem < 100 {
                return format!("{} na {}", m_str, convert_cardinal_integer(rem as i128, use_lakh));
            }
            return format!("{} {}", m_str, convert_cardinal_integer(rem as i128, use_lakh));
        }
    }

    n.to_string()
}

pub fn convert_cardinal_to_swahili(num_str: &str, use_lakh: bool) -> String {
    let mut clean_str = num_str.replace(',', "").trim().to_string();
    let is_negative = if clean_str.starts_with('-') {
        clean_str.remove(0);
        true
    } else {
        false
    };

    if let Some((int_str, frac_str)) = clean_str.split_once('.') {
        if let Ok(int_val) = int_str.parse::<i128>() {
            let mut spoken_int = convert_cardinal_integer(int_val, use_lakh);
            if is_negative {
                spoken_int = format!("hasi {}", spoken_int);
            }
            let decimals: Vec<String> = frac_str
                .chars()
                .filter_map(|c| c.to_digit(10))
                .map(|d| UNITS[d as usize].to_string())
                .collect();
            return format!("{} nukta {}", spoken_int, decimals.join(" "));
        }
    }

    if let Ok(val) = clean_str.parse::<i128>() {
        let spoken = convert_cardinal_integer(val, use_lakh);
        if is_negative {
            format!("hasi {}", spoken)
        } else {
            spoken
        }
    } else {
        num_str.to_string()
    }
}
