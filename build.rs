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

    println!("cargo:rerun-if-env-changed=CHECKER_ENDPOINT");
    match std::env::var("CHECKER_ENDPOINT") {
        Ok(checker_endpoint) => {
            println!(
                "cargo:rustc-env=COMPILED_CHECKER_ENDPOINT={}",
                checker_endpoint
            );
        }
        Err(_) => {
            println!("cargo:rustc-env=COMPILED_CHECKER_ENDPOINT=default-development-endpoint");
        }
    }

    println!("cargo:rerun-if-env-changed=BENCHMARK_ENDPOINT");
    match std::env::var("BENCHMARK_ENDPOINT") {
        Ok(benchmark_endpoint) => {
            println!(
                "cargo:rustc-env=COMPILED_BENCHMARK_ENDPOINT={}",
                benchmark_endpoint
            );
        }
        Err(_) => {
            println!("cargo:rustc-env=COMPILED_BENCHMARK_ENDPOINT=default-development-endpoint");
        }
    }

    println!("cargo:rerun-if-env-changed=SUBMIT_ENDPOINT");
    match std::env::var("SUBMIT_ENDPOINT") {
        Ok(submit_endpoint) => {
            println!(
                "cargo:rustc-env=COMPILED_SUBMIT_ENDPOINT={}",
                submit_endpoint
            );
        }

        Err(_) => {
            println!("cargo:rustc-env=COMPILED_SUBMIT_ENDPOINT=default-development-endpoint");
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
