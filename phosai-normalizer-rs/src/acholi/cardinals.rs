//! Acholi Cardinal Number Verbaliser.

const UNITS: [&str; 10] = [
    "jero", "acel", "aryo", "adek", "angwen", "abic", "abicel", "abiro", "aboro", "abungwen"
];

const TENS: [&str; 10] = [
    "", "apar", "pyere aryo", "pyere adek", "pyere angwen", "pyere abic",
    "pyere abicel", "pyere abiro", "pyere aboro", "pyere abungwen"
];

pub fn convert_cardinal_integer(n: i128) -> String {
    if n < 0 {
        return format!("hasi {}", convert_cardinal_integer(n.abs()));
    }
    let n = n as u128;

    if n < 10 {
        return UNITS[n as usize].to_string();
    }

    if n < 20 {
        if n == 10 {
            return "apar".to_string();
        }
        let rem = (n - 10) as usize;
        return format!("apar wiye {}", UNITS[rem]);
    }

    if n < 100 {
        let ten_val = (n / 10) as usize;
        let rem = (n % 10) as usize;
        if rem == 0 {
            return TENS[ten_val].to_string();
        }
        return format!("{} wiye {}", TENS[ten_val], UNITS[rem]);
    }

    if n < 1000 {
        let hundred_val = (n / 100) as usize;
        let rem = n % 100;
        let h_str = format!("miya {}", UNITS[hundred_val]);
        if rem == 0 {
            return h_str;
        }
        return format!("{} {}", h_str, convert_cardinal_integer(rem as i128));
    }

    if n < 1_000_000 {
        let thousand_val = n / 1000;
        let rem = n % 1000;
        let t_str = format!("alip {}", convert_cardinal_integer(thousand_val as i128));
        if rem == 0 {
            return t_str;
        }
        return format!("{} {}", t_str, convert_cardinal_integer(rem as i128));
    }

    if n < 1_000_000_000 {
        let million_val = n / 1_000_000;
        let rem = n % 1_000_000;
        let m_str = format!("miliyon {}", convert_cardinal_integer(million_val as i128));
        if rem == 0 {
            return m_str;
        }
        return format!("{} {}", m_str, convert_cardinal_integer(rem as i128));
    }

    let billion_val = n / 1_000_000_000;
    let rem = n % 1_000_000_000;
    let b_str = format!("biliyon {}", convert_cardinal_integer(billion_val as i128));
    if rem == 0 {
        return b_str;
    }
    format!("{} {}", b_str, convert_cardinal_integer(rem as i128))
}

pub fn convert_cardinal_to_acholi(num_str: &str) -> String {
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
            return format!("{} nukta {}", spoken_int, decimals.join(" "));
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
