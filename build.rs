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

    println!("cargo:rerun-if-env-changed=GITHUB_CLIENT_ID");
    match std::env::var("GITHUB_CLIENT_ID") {
        Ok(github_client_id) => {
            println!(
                "cargo:rustc-env=COMPILED_GITHUB_CLIENT_ID={}",
                github_client_id
            );
        }
        Err(_) => {
            println!("cargo:rustc-env=COMPILED_GITHUB_CLIENT_ID=default-development-id");
        }
    }

    println!("cargo:rerun-if-env-changed=GITHUB_CLIENT_SECRET");
    match std::env::var("GITHUB_CLIENT_SECRET") {
        Ok(github_client_secret) => {
            println!(
                "cargo:rustc-env=COMPILED_GITHUB_CLIENT_SECRET={}",
                github_client_secret
            );
        }
        Err(_) => {
            println!("cargo:rustc-env=COMPILED_GITHUB_CLIENT_SECRET=default-development-secret");
        }
    }
}
