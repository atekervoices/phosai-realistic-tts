//! PhosAI Normalizer - High-performance Rust text normalizer for African and International languages.

pub mod cleaner;
pub mod luganda;
pub mod swahili;
pub mod kinyarwanda;
pub mod acholi;
pub mod ateso;
pub mod english;

pub use luganda::LugandaNormalizer;
pub use swahili::SwahiliNormalizer;
pub use kinyarwanda::KinyarwandaNormalizer;
pub use acholi::AcholiNormalizer;
pub use ateso::AtesoNormalizer;
pub use english::EnglishNormalizer;

use once_cell::sync::Lazy;

static LUGANDA_NORM: Lazy<LugandaNormalizer> = Lazy::new(LugandaNormalizer::new);
static SWAHILI_NORM: Lazy<SwahiliNormalizer> = Lazy::new(SwahiliNormalizer::new);
static KINYARWANDA_NORM: Lazy<KinyarwandaNormalizer> = Lazy::new(KinyarwandaNormalizer::new);
static ACHOLI_NORM: Lazy<AcholiNormalizer> = Lazy::new(AcholiNormalizer::new);
static ATESO_NORM: Lazy<AtesoNormalizer> = Lazy::new(AtesoNormalizer::new);
static ENGLISH_NORM: Lazy<EnglishNormalizer> = Lazy::new(EnglishNormalizer::new);

/// Automatically extracts the language code from a TTS speaker ID or voice name.
///
/// Examples:
/// - "lug_female_4", "luganda_female", "lug_male_1" -> "lug"
/// - "swa_female_1", "sw_male_2" -> "swa"
/// - "kin_female_1", "rw_male_1" -> "kin"
/// - "ach_female_2", "ach_male_1" -> "ach"
/// - "teo_female_1", "ate_male_1" -> "teo"
/// - "eng_female_1", "pcm_female_1" -> "eng"
/// - "nyn_female_1", "xog_female_1" -> "lug"
pub fn detect_language_from_speaker(speaker_id: &str) -> &'static str {
    let lower = speaker_id.trim().to_lowercase();
    if lower.starts_with("lug") {
        "lug"
    } else if lower.starts_with("swa") || lower.starts_with("sw_") || lower == "sw" {
        "swa"
    } else if lower.starts_with("kin") || lower.starts_with("rw_") || lower == "rw" {
        "kin"
    } else if lower.starts_with("ach") {
        "ach"
    } else if lower.starts_with("teo") || lower.starts_with("ate") {
        "teo"
    } else if lower.starts_with("eng") || lower.starts_with("en_") || lower.starts_with("pcm") {
        "eng"
    } else if lower.starts_with("nyn") || lower.starts_with("xog") {
        "lug"
    } else {
        // Default fallback to Luganda
        "lug"
    }
}

/// Main entry point for TTS text normalisation.
///
/// Parameters:
/// - `text`: Raw text with Non-Standard Words (NSWs) like numbers, currencies, dates, times, symbols.
/// - `lang`: 3-letter or 2-letter ISO language code ("lug", "swa"/"sw", "kin"/"rw", "ach", "teo"/"ate", "eng"/"en").
pub fn normalize(text: &str, lang: &str) -> String {
    let preprocessed = cleaner::fix_glued_punctuation(text);
    let code = lang.trim().to_lowercase();
    let result = match code.as_str() {
        "lug" | "luganda" | "nyn" | "xog" => LUGANDA_NORM.normalize(&preprocessed),
        "swa" | "sw" | "swahili" | "kiswahili" => SWAHILI_NORM.normalize(&preprocessed),
        "kin" | "rw" | "kinyarwanda" => KINYARWANDA_NORM.normalize(&preprocessed),
        "ach" | "acholi" => ACHOLI_NORM.normalize(&preprocessed),
        "teo" | "ate" | "ateso" => ATESO_NORM.normalize(&preprocessed),
        "eng" | "en" | "english" | "pcm" => ENGLISH_NORM.normalize(&preprocessed),
        _ => {
            // If unknown, fallback to Luganda or return original cleaned text
            LUGANDA_NORM.normalize(&preprocessed)
        }
    };
    cleaner::sanitize_tts_post(&result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_luganda_pipeline() {
        let res = normalize("Musoke, ye muddusi 24 m, obudde 07:00. Baasasudde UGX 50,000.", "lug");
        assert!(res.contains("mmita amakumi abiri mu nnya"), "Result was: {}", res);
        assert!(res.contains("ssaawa emu ez'enkya"), "Result was: {}", res);
        assert!(res.contains("Ssiringi za Uganda emitwaalo etaano"), "Result was: {}", res);
    }

    #[test]
    fn test_luganda_noun_class_agreement() {
        assert!(normalize("24 m", "lug").contains("mmita amakumi abiri mu nnya"));
        assert!(normalize("Abantu 24", "lug").contains("Abantu amakumi abiri mu bana"));
        assert!(normalize("emyaka 24", "lug").contains("emyaka amakumi abiri mu ana"));
    }

    #[test]
    fn test_luganda_currencies() {
        assert!(normalize("UGX 50,000", "lug").contains("Ssiringi za Uganda emitwaalo etaano"));
        assert!(normalize("100.50/=", "lug").contains("Ssiringi kikumi n'ennusu amakumi ataano"));
        assert!(normalize("$50", "lug").contains("Ddoola z'Amerika amakumi ataano"));
        assert!(normalize("$2.50", "lug").contains("Ddoola z'Amerika bbiri n'ennusu amakumi ataano"));
        assert!(normalize("£50", "lug").contains("Pawundi amakumi ataano"));
    }

    #[test]
    fn test_luganda_phone_and_abbrevs() {
        let res = normalize("070278319", "lug");
        assert!(res.contains("zeero musanvu zeero bbiri musanvu munaana ssatu emu mwenda"));
        assert!(normalize("Mw. Mugisha", "lug").contains("Mwami Mugisha"));
        assert!(normalize("URA", "lug").contains("Uganda Revenue Authority"));
    }

    #[test]
    fn test_swahili_pipeline() {
        let res = normalize("Bw. Juma alilipa KES 50,000 tarehe 16/07/2026 saa 07:00.", "swa");
        assert!(res.contains("Bwana Juma"), "Got: {}", res);
        assert!(res.contains("shilingi za kenya elfu hamsini"), "Got: {}", res);
        assert!(res.contains("tarehe kumi na sita mwezi wa saba mwaka elfu mbili na ishirini na sita"), "Got: {}", res);
        assert!(res.contains("saa moja asubuhi"), "Got: {}", res);

        let res_sw = normalize("Bi. Asha ana 10 kg.", "sw");
        assert!(res_sw.contains("Bibi Asha"));
        assert!(res_sw.contains("kilogramu kumi"));
    }

    #[test]
    fn test_swahili_lakh_mode() {
        assert!(normalize("100000", "swa").contains("laki moja"));
        assert!(normalize("KES 200,000", "swa").contains("shilingi za kenya laki mbili"));
    }

    #[test]
    fn test_kinyarwanda_pipeline() {
        let res = normalize("Bw. Kagabo yasabye RWF 2,000 ku itariki 16/07/2026 saa 07:00.", "kin");
        assert!(res.contains("Bwana Kagabo"), "Got: {}", res);
        assert!(res.contains("amafaranga y'u rwanda ibihumbi bibiri"), "Got: {}", res);
        assert!(res.contains("ku itariki cumi na gatandatu ukwa karindwi umwaka w'ibihumbi bibiri na makumyabiri na gatandatu"), "Got: {}", res);
        assert!(res.contains("saa moya mu mugitondo"), "Got: {}", res);

        let res_rw = normalize("Bi. Mukamana afite 10 kg.", "rw");
        assert!(res_rw.contains("Bibi Mukamana"));
        assert!(res_rw.contains("ibilo cumi"));
        assert!(normalize("RBC", "kin").contains("arabisi"));
        assert!(normalize("RISA", "kin").contains("risa"));
    }

    #[test]
    fn test_acholi_pipeline() {
        let res = normalize("Okello tye ki 10 kg i cawa 07:00. Chwalo UGX 50,000.", "ach");
        assert!(res.contains("kilogram apar"), "Got: {}", res);
        assert!(res.contains("cawa acel me odiko"), "Got: {}", res);
        assert!(res.contains("Shilling me Uganda alip pyere abic"), "Got: {}", res);
        assert!(normalize("123", "ach").contains("miya acel pyere aryo wiye adek"));
        assert!(normalize("1st", "ach").contains("Me acel"));
    }

    #[test]
    fn test_ateso_pipeline() {
        let res = normalize("Omodok eyait 10 kg kak esaa 07:00.", "teo");
        assert!(res.contains("ekilogram itomon"), "Got: {}", res);
        assert!(res.contains("esaa ediopet taparach"), "Got: {}", res);
        assert!(normalize("1st", "teo").contains("losodit"));
        assert!(normalize("2nd", "teo").contains("lokiareit"));
        assert!(normalize("UGX 50,000", "teo").contains("Esurupi ka Uganda ilukumin akais-akany"));
    }

    #[test]
    fn test_english_pipeline() {
        let res = normalize("Dr. Mugisha paid $50.25 on 12/04/2026 at 08:30 AM.", "eng");
        assert!(res.contains("Doctor Mugisha"), "Got: {}", res);
        assert!(res.contains("fifty dollars and twenty five cents"), "Got: {}", res);
        assert!(res.contains("April twelfth, twenty twenty six"), "Got: {}", res);
        assert!(res.contains("eight thirty A M"), "Got: {}", res);
        
        let res2 = normalize("Musoke paid UGX 50,000 on 16/07/2026 at 07:00.", "eng");
        assert!(res2.contains("fifty thousand Uganda shillings"), "Got: {}", res2);
        assert!(res2.contains("July sixteenth, twenty twenty six"), "Got: {}", res2);
        assert!(res2.contains("seven o'clock A M"), "Got: {}", res2);

        assert!(normalize("123", "eng").contains("one hundred twenty three"));
        assert!(normalize("22nd", "eng").contains("twenty second"));

        // Comma separated numbers & Academic year ranges
        let num_test = normalize("We received 3,664 Applications; 2,904 Undergraduates and 760 master's students. 1,440 admitted for the 2026/2027 academic year.", "eng");
        assert!(num_test.contains("three thousand six hundred sixty four"), "Got: {}", num_test);
        assert!(num_test.contains("two thousand nine hundred four"), "Got: {}", num_test);
        assert!(num_test.contains("seven hundred sixty"), "Got: {}", num_test);
        assert!(num_test.contains("one thousand four hundred forty"), "Got: {}", num_test);
        assert!(num_test.contains("twenty twenty six to twenty twenty seven") || num_test.contains("two thousand twenty six to two thousand twenty seven"), "Got: {}", num_test);

        let lug_num_test = normalize("Abayizi 3,664; 2,904 ne 1,440 mu mwaka gwa 2026/2027.", "lug");
        assert!(lug_num_test.contains("nkumi ssatu mu lukaaga mu nkaaga mu bana"), "Got: {}", lug_num_test);
        assert!(lug_num_test.contains("nkumi bbiri mu lwenda mu nnya"), "Got: {}", lug_num_test);
        assert!(lug_num_test.contains("lukumi mu bina mu amakumi ana"), "Got: {}", lug_num_test);
        assert!(lug_num_test.contains("nkumi bbiri mu amakumi abiri mu mukaaga okutuuka ku nkumi bbiri mu amakumi abiri mu musanvu"), "Got: {}", lug_num_test);
    }

    #[test]
    fn test_speaker_language_detection() {
        assert_eq!(detect_language_from_speaker("lug_female_4"), "lug");
        assert_eq!(detect_language_from_speaker("luganda_female"), "lug");
        assert_eq!(detect_language_from_speaker("swa_male_3"), "swa");
        assert_eq!(detect_language_from_speaker("kin_female_1"), "kin");
        assert_eq!(detect_language_from_speaker("ach_female_2"), "ach");
        assert_eq!(detect_language_from_speaker("teo_female_1"), "teo");
        assert_eq!(detect_language_from_speaker("eng_female_1"), "eng");
        assert_eq!(detect_language_from_speaker("pcm_female_1"), "eng");
        assert_eq!(detect_language_from_speaker("nyn_male_3"), "lug");
        assert_eq!(detect_language_from_speaker("xog_female_1"), "lug");
    }
}
