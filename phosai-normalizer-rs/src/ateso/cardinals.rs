//! Ateso Cardinal Number Verbaliser.

const UNITS: [&str; 10] = [
    "jero", "ediopet", "iarei", "iuni", "ioangon", "ikany", "ikanyape", "ikanyaarei", "ikanyauni", "ikanyaongon"
];

const TEENS: [&str; 10] = [
    "itomon", "itomon'k'adiope", "itomon'aarei", "itomon'auni", "itomon'aaŋon",
    "itomon'akany", "itomon akanyape", "itomon akanyaarei", "itomon akanyauni", "itomon akanyaongon"
];

const TENS: [&str; 10] = [
    "", "itomon", "akais-aarei", "akais-auni", "akais-aongon", "akais-akany",
    "akais-akanyape", "akais-akanyaarei", "akais-akanyauni", "akais-akanyaongon"
];

pub fn convert_cardinal_integer(n: i128) -> String {
    if n < 0 {
        return format!("hasi {}", convert_cardinal_integer(n.abs()));
    }
    let n = n as u128;

    if n < 10 {
        return UNITS[n as usize].to_string();
    }

    if (10..20).contains(&n) {
        return TEENS[(n - 10) as usize].to_string();
    }

    if n < 100 {
        let ten_val = (n / 10) as usize;
        let rem = (n % 10) as usize;
        if rem == 0 {
            return TENS[ten_val].to_string();
        }
        return format!("{} ka {}", TENS[ten_val], UNITS[rem]);
    }

    if n < 1000 {
        if n == 100 {
            return "akwatat".to_string();
        }
        let hundred_val = (n / 100) as usize;
        let rem = n % 100;
        let h_str = format!("akwat {}", UNITS[hundred_val]);
        if rem == 0 {
            return h_str;
        }
        return format!("{} ka {}", h_str, convert_cardinal_integer(rem as i128));
    }

    if n < 1_000_000 {
        let thousand_val = n / 1000;
        let rem = n % 1000;
        let t_str = if thousand_val == 1 {
            "elukumit ediope".to_string()
        } else {
            format!("ilukumin {}", convert_cardinal_integer(thousand_val as i128))
        };
        if rem == 0 {
            return t_str;
        }
        return format!("{} ka {}", t_str, convert_cardinal_integer(rem as i128));
    }

    if n < 1_000_000_000 {
        let million_val = n / 1_000_000;
        let rem = n % 1_000_000;
        let m_str = if million_val == 1 {
            "emilionit ediope".to_string()
        } else {
            format!("imilionin {}", convert_cardinal_integer(million_val as i128))
        };
        if rem == 0 {
            return m_str;
        }
        return format!("{} ka {}", m_str, convert_cardinal_integer(rem as i128));
    }

    let billion_val = n / 1_000_000_000;
    let rem = n % 1_000_000_000;
    let b_str = format!("imilionin akwatat {}", convert_cardinal_integer(billion_val as i128));
    if rem == 0 {
        return b_str;
    }
    format!("{} ka {}", b_str, convert_cardinal_integer(rem as i128))
}

pub fn convert_cardinal_to_ateso(num_str: &str) -> String {
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
