fn main() {
    #[cfg(target_os = "linux")]
    {
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
        println!("cargo:rustc-link-search=native=/app");
        println!("cargo:rustc-link-lib=static=ort_compat");
    }
}

