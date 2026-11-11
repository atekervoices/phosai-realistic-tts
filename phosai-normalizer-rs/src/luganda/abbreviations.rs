//! Luganda Abbreviation and Acronym Expansion.

use regex::Regex;
use once_cell::sync::Lazy;

const ABBREVIATIONS: &[(&str, &str)] = &[
    ("Mw.", "Mwami"),
    ("Mw", "Mwami"),
    ("Owek.", "Oweekitiibwa"),
    ("Owek", "Oweekitiibwa"),
    ("Dr.", "Dokita"),
    ("Dr", "Dokita"),
    ("Prof.", "Purofesa"),
    ("Prof", "Purofesa"),
    ("FM", "ffa mma"),
    ("DP", "ddi ppi"),
    ("e.g.", "gamba nga"),
    ("e.g", "gamba nga"),
    ("etc.", "n'ebirala"),
    ("etc", "n'ebirala"),
    ("i.e.", "kutegeeza"),
    ("i.e", "kutegeeza"),
    ("URA", "Uganda Revenue Authority"),
    ("KCCA", "Kampala Capital City Authority"),
    ("UNEB", "Uganda National Examinations Board"),
    ("NSSF", "National Social Security Fund"),
    ("UPDF", "Uganda People's Defence Forces"),
    ("UPF", "Uganda Police Force"),
    ("NRM", "National Resistance Movement"),
    ("UBC", "Uganda Broadcasting Corporation"),
    ("UNRA", "Uganda National Roads Authority"),
    ("URSB", "Uganda Registration Services Bureau"),
    ("UIA", "Uganda Investment Authority"),
    ("UCC", "Uganda Communications Commission"),
    ("ULC", "Uganda Land Commission"),
    ("URC", "Uganda Railways Corporation"),
    ("NWSC", "National Water and Sewerage Corporation"),
    ("UEDCL", "Uganda Electricity Distribution Company Limited"),
    ("UEGCL", "Uganda Electricity Generation Company Limited"),
    ("UETCL", "Uganda Electricity Transmission Company Limited"),
    ("ERA", "Electricity Regulatory Authority"),
    ("NEMA", "National Environment Management Authority"),
    ("NFA", "National Forestry Authority"),
    ("UWA", "Uganda Wildlife Authority"),
    ("UTB", "Uganda Tourism Board"),
    ("UNBS", "Uganda National Bureau of Standards"),
    ("BOU", "Bank of Uganda"),
    ("PPDA", "Public Procurement and Disposal of Public Assets Authority"),
    ("IGG", "Inspectorate of Government"),
    ("DPP", "Directorate of Public Prosecutions"),
    ("ISO", "Internal Security Organisation"),
    ("ESO", "External Security Organisation"),
    ("UHRC", "Uganda Human Rights Commission"),
    ("EC", "Electoral Commission"),
    ("NIRA", "National Identification and Registration Authority"),
    ("NITA-U", "National Information Technology Authority - Uganda"),
    ("UBOS", "Uganda Bureau of Statistics"),
    ("UDB", "Uganda Development Bank"),
    ("UDC", "Uganda Development Corporation"),
    ("UIC", "Uganda Insurance Commission"),
    ("UNCST", "Uganda National Council for Science and Technology"),
    ("UNMA", "Uganda National Meteorological Authority"),
    ("UMA", "Uganda Manufacturers' Association"),
    ("UNOC", "Uganda National Oil Company"),
    ("URF", "Uganda Road Fund"),
    ("UIRI", "Uganda Industrial Research Institute"),
    ("ULRC", "Uganda Law Reform Commission"),
    ("NAADS", "National Agricultural Advisory Services"),
    ("NARO", "National Agricultural Research Organisation"),
    ("NPA", "National Planning Authority"),
    ("NCHE", "National Council for Higher Education"),
    ("UCAA", "Uganda Civil Aviation Authority"),
    ("OPM", "Office of the Prime Minister"),
    ("UPE", "Universal Primary Education"),
    ("USE", "Universal Secondary Education"),
    ("MUST", "Mbarara University of Science and Technology"),
    ("MAK", "Makerere University"),
    ("KYU", "Kyambogo University"),
    ("UMI", "Uganda Management Institute"),
    ("UPC", "Uganda People's Congress"),
    ("FDC", "Forum for Democratic Change"),
    ("NUP", "National Unity Platform"),
];

static PATTERNS: Lazy<Vec<(Regex, &'static str)>> = Lazy::new(|| {
    ABBREVIATIONS
        .iter()
        .map(|(abbrev, expansion)| {
            let pat = format!(r"(^|[^\w]){}([^\w]|$)", regex::escape(abbrev));
            (Regex::new(&pat).unwrap(), *expansion)
        })
        .collect()
});

pub fn replace_abbreviations_in_text(mut text: String) -> String {
    for (re, expansion) in PATTERNS.iter() {
        text = re.replace_all(&text, |caps: &regex::Captures| {
            let prefix = caps.get(1).map_or("", |m| m.as_str());
            let suffix = caps.get(2).map_or("", |m| m.as_str());
            format!("{}{}{}", prefix, expansion, suffix)
        }).into_owned();
    }
    text
}
