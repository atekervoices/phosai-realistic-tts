//! Text cleaning and unicode standardization utilities.

use regex::Regex;
use unicode_normalization::UnicodeNormalization;
use once_cell::sync::Lazy;

static SPACE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[^\S\n]+").unwrap());
static MULTI_SPACE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[ ]{2,}").unwrap());
static TAB_CR_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[\r\t]").unwrap());

/// Normalize string to Unicode NFC form and replace non-standard spaces/control chars.
pub fn normalize_unicode(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    let nfc: String = text.nfc().collect();
    let text = TAB_CR_RE.replace_all(&nfc, " ");
    let text = SPACE_RE.replace_all(&text, " ");
    text.into_owned()
}

/// Standardizes smart quotes, typography dashes, and ellipses.
pub fn standardize_punctuation(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            // Smart single quotes & modifier apostrophes
            '\u{2018}' | '\u{2019}' | '\u{201B}' | '\u{02BC}' | '`' | '´' => out.push('\''),
            // Smart double quotes
            '\u{201C}' | '\u{201D}' | '\u{201E}' | '\u{201F}' | '«' | '»' => out.push('"'),
            // Typography dashes
            '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' | '\u{2015}' | '−' => out.push('-'),
            // Ellipsis
            '\u{2026}' => out.push_str("..."),
            _ => out.push(c),
        }
    }
    out
}

/// Normalize mathematical alphanumeric symbols (U+1D400 to U+1D7FF) to standard ASCII.
pub fn normalize_math_symbols(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        let cp = c as u32;
        if (0x1D400..=0x1D7FF).contains(&cp) {
            let mapped = match cp {
                // Bold Capital
                0x1D400..=0x1D419 => char::from_u32('A' as u32 + (cp - 0x1D400)),
                // Bold Small
                0x1D41A..=0x1D433 => char::from_u32('a' as u32 + (cp - 0x1D41A)),
                // Italic Capital
                0x1D434..=0x1D44D => char::from_u32('A' as u32 + (cp - 0x1D434)),
                // Italic Small
                0x1D44E..=0x1D467 => char::from_u32('a' as u32 + (cp - 0x1D44E)),
                // Bold Italic Capital
                0x1D468..=0x1D481 => char::from_u32('A' as u32 + (cp - 0x1D468)),
                // Bold Italic Small
                0x1D482..=0x1D49B => char::from_u32('a' as u32 + (cp - 0x1D482)),
                // Script Capital
                0x1D49C..=0x1D4B5 => char::from_u32('A' as u32 + (cp - 0x1D49C)),
                // Script Small
                0x1D4B6..=0x1D4CF => char::from_u32('a' as u32 + (cp - 0x1D4B6)),
                // Fraktur Capital
                0x1D504..=0x1D51D => char::from_u32('A' as u32 + (cp - 0x1D504)),
                // Fraktur Small
                0x1D51E..=0x1D537 => char::from_u32('a' as u32 + (cp - 0x1D51E)),
                // Double-struck Capital
                0x1D538..=0x1D551 => char::from_u32('A' as u32 + (cp - 0x1D538)),
                // Double-struck Small
                0x1D552..=0x1D56B => char::from_u32('a' as u32 + (cp - 0x1D552)),
                // Sans-serif Capital
                0x1D5A0..=0x1D5B9 => char::from_u32('A' as u32 + (cp - 0x1D5A0)),
                // Sans-serif Small
                0x1D5BA..=0x1D5D3 => char::from_u32('a' as u32 + (cp - 0x1D5BA)),
                // Sans-serif Bold Capital
                0x1D5D4..=0x1D5ED => char::from_u32('A' as u32 + (cp - 0x1D5D4)),
                // Sans-serif Bold Small
                0x1D5EE..=0x1D607 => char::from_u32('a' as u32 + (cp - 0x1D5EE)),
                // Sans-serif Bold Italic Capital
                0x1D608..=0x1D621 => char::from_u32('A' as u32 + (cp - 0x1D608)),
                // Sans-serif Bold Italic Small
                0x1D622..=0x1D63B => char::from_u32('a' as u32 + (cp - 0x1D622)),
                // Monospace Capital
                0x1D670..=0x1D689 => char::from_u32('A' as u32 + (cp - 0x1D670)),
                // Monospace Small
                0x1D68A..=0x1D6A3 => char::from_u32('a' as u32 + (cp - 0x1D68A)),
                // Bold Digits
                0x1D7CE..=0x1D7D7 => char::from_u32('0' as u32 + (cp - 0x1D7CE)),
                // Double-struck Digits
                0x1D7D8..=0x1D7E1 => char::from_u32('0' as u32 + (cp - 0x1D7D8)),
                // Sans-serif Digits
                0x1D7E2..=0x1D7EB => char::from_u32('0' as u32 + (cp - 0x1D7E2)),
                // Sans-serif Bold Digits
                0x1D7EC..=0x1D7F5 => char::from_u32('0' as u32 + (cp - 0x1D7EC)),
                // Monospace Digits
                0x1D7F6..=0x1D7FF => char::from_u32('0' as u32 + (cp - 0x1D7F6)),
                _ => None,
            };
            if let Some(mc) = mapped {
                out.push(mc);
            } else {
                out.push(c);
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Strip emojis or replace symbols that interfere with TTS pronunciation.
pub fn clean_emojis(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        let cp = c as u32;
        // Skip common emoji ranges
        if (0x1F600..=0x1F64F).contains(&cp) // Emoticons
            || (0x1F300..=0x1F5FF).contains(&cp) // Misc Symbols and Pictographs
            || (0x1F680..=0x1F6FF).contains(&cp) // Transport and Map
            || (0x1F700..=0x1F77F).contains(&cp) // Alchemical Symbols
            || (0x1F780..=0x1F7FF).contains(&cp) // Geometric Shapes Extended
            || (0x1F800..=0x1F8FF).contains(&cp) // Supplemental Arrows-C
            || (0x1F900..=0x1F9FF).contains(&cp) // Supplemental Symbols and Pictographs
            || (0x1FA00..=0x1FA6F).contains(&cp) // Chess Symbols
            || (0x1FA70..=0x1FAFF).contains(&cp) // Symbols and Pictographs Extended-A
            || (0x2600..=0x26FF).contains(&cp)   // Misc symbols (☀️, ☔, etc.)
            || (0x2700..=0x27BF).contains(&cp)   // Dingbats
            || (0xFE00..=0xFE0F).contains(&cp)   // Variation Selectors
            || (0x1F1E6..=0x1F1FF).contains(&cp) // Flags
        {
            out.push(' ');
        } else {
            out.push(c);
        }
    }
    out
}

/// Collapse multiple spaces into a single space and strip leading/trailing whitespace.
pub fn clean_whitespace(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }
    let res = MULTI_SPACE_RE.replace_all(text, " ");
    res.trim().to_string()
}

/// Separates punctuation that was glued to words or numbers without a space (e.g. from LLM generation).
pub fn fix_glued_punctuation(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }

    // 1. Convert typography em-dash and en-dash to comma with space
    let mut s = text.replace('—', ", ").replace('–', ", ").replace(" - ", ", ");

    // 2. Replace ellipsis (...) with single period
    static ELLIPSIS_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\.{2,}").unwrap());
    s = ELLIPSIS_RE.replace_all(&s, ".").into_owned();

    // 3. Separate glued punctuation (missing spaces from LLM outputs):
    // Letter.Letter -> Letter. Letter (e.g. "needs.If" -> "needs. If")
    static DOT_LETTER_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"([a-zA-Z])\.([a-zA-Z])").unwrap());
    s = DOT_LETTER_RE.replace_all(&s, "$1. $2").into_owned();

    // Letter.Digit -> Letter. Digit (e.g. "systems.4." -> "systems. 4.")
    static DOT_DIGIT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"([a-zA-Z])\.(\d)").unwrap());
    s = DOT_DIGIT_RE.replace_all(&s, "$1. $2").into_owned();

    // Digit.Letter -> Digit. Letter (e.g. "23.I'll" -> "23. I'll")
    static DIGIT_DOT_LETTER_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(\d)\.([a-zA-Z])").unwrap());
    s = DIGIT_DOT_LETTER_RE.replace_all(&s, "$1. $2").into_owned();

    // Letter:Digit -> Letter: Digit (e.g. "include:1." -> "include: 1.")
    static COLON_DIGIT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"([a-zA-Z]):(\d)").unwrap());
    s = COLON_DIGIT_RE.replace_all(&s, "$1: $2").into_owned();

    // Letter:Letter -> Letter: Letter (e.g. "Note:please" -> "Note: please")
    static COLON_LETTER_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"([a-zA-Z]):([a-zA-Z])").unwrap());
    s = COLON_LETTER_RE.replace_all(&s, "$1: $2").into_owned();

    // Letter,Letter -> Letter, Letter (e.g. "yes,sir" -> "yes, sir")
    static COMMA_LETTER_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"([a-zA-Z]),([a-zA-Z])").unwrap());
    s = COMMA_LETTER_RE.replace_all(&s, "$1, $2").into_owned();

    s
}

/// Post-normalization sanitizer for TTS audio clarity:
/// - Replaces any residual colons (times are already verbalized to words) with clean breaks/pauses
/// - Cleans duplicate punctuation, space before punctuation, etc.
pub fn sanitize_tts_post(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }

    let mut s = text.to_string();

    // 1. Colon followed by newline -> period and newline
    static COLON_NEWLINE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r":\s*(\n+)").unwrap());
    s = COLON_NEWLINE_RE.replace_all(&s, ".$1").into_owned();

    // 2. Colon followed by capital letter -> comma, space, and lowercase letter for natural speech flow
    // e.g. "Solar Energy Systems: We design" -> "Solar Energy Systems, we design"
    static COLON_CAPITAL_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r":\s*([A-Z])").unwrap());
    s = COLON_CAPITAL_RE.replace_all(&s, |caps: &regex::Captures| {
        let lower = caps[1].to_lowercase();
        format!(", {}", lower)
    }).into_owned();

    // 3. Any remaining residual colon -> comma and space
    // e.g. "offerings include: 1." -> "offerings include, 1."
    static ANY_COLON_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r":\s*").unwrap());
    s = ANY_COLON_RE.replace_all(&s, ", ").into_owned();

    // Clean up duplicate or messy punctuation:
    static DUP_PUNCT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[,;][,;\s]*,").unwrap());
    s = DUP_PUNCT_RE.replace_all(&s, ",").into_owned();

    static COMMA_DOT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r",\s*\.").unwrap());
    s = COMMA_DOT_RE.replace_all(&s, ".").into_owned();

    static DOT_COMMA_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\.\s*,").unwrap());
    s = DOT_COMMA_RE.replace_all(&s, ".").into_owned();

    // Space before comma or period -> remove space
    static SPACE_BEFORE_PUNCT: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+([,.!?])").unwrap());
    s = SPACE_BEFORE_PUNCT.replace_all(&s, "$1").into_owned();

    // Ensure comma always followed by space if followed by a word
    static COMMA_NO_SPACE: Lazy<Regex> = Lazy::new(|| Regex::new(r",([^\s\d])").unwrap());
    s = COMMA_NO_SPACE.replace_all(&s, ", $1").into_owned();

    s
}

