fn main() {
    println!("cargo:rerun-if-env-changed=MODAL_SLUG");
    match std::env::var("MODAL_SLUG") {
        Ok(modal_slug) => {
            println!("cargo:rustc-env=COMPILED_MODAL_SLUG={}", modal_slug);
        }
        Err(_) => {
            println!("cargo:rustc-env=COMPILED_MODAL_SLUG=default-development-slug");
        }
    }
}
