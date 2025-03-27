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

    println!("cargo:rerun-if-env-changed=CLIENT_ID");
    match std::env::var("CLIENT_ID") {
        Ok(client_id) => {
            println!("cargo:rustc-env=COMPILED_CLIENT_ID={}", client_id);
        }
        Err(_) => {
            println!("cargo:rustc-env=COMPILED_CLIENT_ID=default-development-id");
        }
    }

    println!("cargo:rerun-if-env-changed=CLIENT_SECRET");
    match std::env::var("CLIENT_SECRET") {
        Ok(client_secret) => {
            println!("cargo:rustc-env=COMPILED_CLIENT_SECRET={}", client_secret);
        }
        Err(_) => {
            println!("cargo:rustc-env=COMPILED_CLIENT_SECRET=default-development-secret");
        }
    }

    println!("cargo:rerun-if-env-changed=PROBLEM_ENDPOINT");
    match std::env::var("PROBLEM_ENDPOINT") {
        Ok(problem_endpoint) => {
            println!("cargo:rustc-env=COMPILED_PROBLEM_URL={}", problem_endpoint);
        }
        Err(_) => {
            println!("cargo:rustc-env=COMPILED_PROBLEM_URL=default-development-endpoint");
        }
    }
}
