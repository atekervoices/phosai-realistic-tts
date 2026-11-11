//! Kinyarwanda Cardinal Number Verbaliser.

const UNITS: [&str; 10] = [
    "zero", "rimwe", "kabiri", "gatatu", "kane", "gatanu", "gatandatu", "karindwi", "umunani", "icyenda"
];

const TENS: [&str; 10] = [
    "", "cumi", "makumyabiri", "mirongo itatu", "mirongo ine", "mirongo itanu",
    "mirongo itandatu", "mirongo arindwi", "mirongo umunani", "mirongo cyenda"
];

const HUNDREDS: [&str; 10] = [
    "", "ijana", "magana abiri", "magana atatu", "magana ane", "magana atanu",
    "magana atandatu", "magana arindwi", "magana umunani", "magana cyenda"
];

const THOUSANDS_AGREEMENT: [&str; 10] = [
    "", "igihumbi", "ibihumbi bibiri", "ibihumbi bitatu", "ibihumbi bine",
    "ibihumbi bitanu", "ibihumbi bitandatu", "ibihumbi barindwi", "ibihumbi umunani", "ibihumbi cyenda"
];

fn join_na(first: &str, second: &str) -> String {
    if second.starts_with(['u', 'i', 'a', 'e', 'o']) {
        format!("{} n'{}", first, second)
    } else {
        format!("{} na {}", first, second)
    }
}

pub fn convert_cardinal_integer(n: i128) -> String {
    if n < 0 {
        return format!("hasi {}", convert_cardinal_integer(n.abs()));
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
        return join_na(TENS[ten_val], UNITS[rem]);
    }

    if n < 1000 {
        let hundred_val = (n / 100) as usize;
        let rem = (n % 100) as usize;
        let h_str = HUNDREDS[hundred_val];
        if rem == 0 {
            return h_str.to_string();
        }
        return join_na(h_str, &convert_cardinal_integer(rem as i128));
    }

    if n < 1_000_000 {
        let thousand_val = (n / 1000) as usize;
        let rem = (n % 1000) as usize;
        let t_str = if thousand_val < 10 {
            THOUSANDS_AGREEMENT[thousand_val].to_string()
        } else {
            format!("ibihumbi {}", convert_cardinal_integer(thousand_val as i128))
        };
        if rem == 0 {
            return t_str;
        }
        return join_na(&t_str, &convert_cardinal_integer(rem as i128));
    }

    if n < 1_000_000_000 {
        let million_val = n / 1_000_000;
        let rem = n % 1_000_000;
        let m_str = if million_val == 1 {
            "miliyoni imwe".to_string()
        } else {
            format!("miliyoni {}", convert_cardinal_integer(million_val as i128))
        };
        if rem == 0 {
            return m_str;
        }
        return join_na(&m_str, &convert_cardinal_integer(rem as i128));
    }

    if n < 1_000_000_000_000 {
        let billion_val = n / 1_000_000_000;
        let rem = n % 1_000_000_000;
        let b_str = if billion_val == 1 {
            "miliyari imwe".to_string()
        } else {
            format!("miliyari {}", convert_cardinal_integer(billion_val as i128))
        };
        if rem == 0 {
            return b_str;
        }
        return join_na(&b_str, &convert_cardinal_integer(rem as i128));
    }

    let trillion_val = n / 1_000_000_000_000;
    let rem = n % 1_000_000_000_000;
    let tr_str = if trillion_val == 1 {
        "triliyoni imwe".to_string()
    } else {
        format!("triliyoni {}", convert_cardinal_integer(trillion_val as i128))
    };
    if rem == 0 {
        return tr_str;
    }
    join_na(&tr_str, &convert_cardinal_integer(rem as i128))
}

pub fn convert_cardinal_to_kinyarwanda(num_str: &str) -> String {
    let mut clean_str = num_str.replace(',', "").trim().to_string();
    let is_negative = if clean_str.starts_with('-') {
        clean_str.remove(0);
        true
    } else {
        false
    };

    if let Some((int_str, frac_str)) = clean_str.split_once('.') {
        if let Ok(int_val) = int_str.parse::<i128>() {
            let mut spoken_int = convert_cardinal_integer(int_val);
            if is_negative {
                spoken_int = format!("hasi {}", spoken_int);
            }
            let decimals: Vec<String> = frac_str
                .chars()
                .filter_map(|c| c.to_digit(10))
                .map(|d| UNITS[d as usize].to_string())
                .collect();
            return format!("{} nukuta {}", spoken_int, decimals.join(" "));
        }
    }

    if let Ok(val) = clean_str.parse::<i128>() {
        let spoken = convert_cardinal_integer(val);
        if is_negative {
            format!("hasi {}", spoken)
        } else {
            spoken
        }
    } else {
        num_str.to_string()
    }
}
