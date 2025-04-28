use dotenv::dotenv;
use tensara::{
    auth::{ensure_authenticated, ensure_authenticated_next},
    client, pretty::{self, pretty_print_problems},
    trpc::*,
    Parameters,
}; // Add this to the top if not already

const COMPILED_MODAL_SLUG: &str = env!("COMPILED_MODAL_SLUG");

fn main() {
    #[cfg(debug_assertions)]
    dotenv().ok();
    let auth_info = ensure_authenticated();

    let username = &auth_info.github_username;
    let parameters: Parameters = Parameters::new(Some(username.clone()));

    let command_type = parameters.get_command_name();
    let dtype = parameters.get_dtype();
    let gpu_type = parameters.get_gpu_type();
    let problem_def = parameters.get_problem_def();
    let problem = parameters.get_problem();
    let language = parameters.get_language();

    let modal_slug =
        std::env::var("MODAL_SLUG").unwrap_or_else(|_| COMPILED_MODAL_SLUG.to_string());
    let endpoint = format!("{}/{}-{}", modal_slug, command_type, gpu_type);
    let endpoint = endpoint.as_str();

    let response = client::send_post_request(
        endpoint,
        &parameters.get_solution_code(),
        &problem,
        &problem_def,
        &dtype,
        &language,
    );

    match command_type.as_str() {
        "benchmark" => pretty::pretty_print_benchmark_response(response),
        "checker" => pretty::pretty_print_checker_streaming_response(response),
        "submit" => {
            if ensure_authenticated_next() {
                println!("Auth successful....");

                let problem_slug = parameters.get_problem();
                let code = parameters.get_solution_code();
                let language = parameters.get_language();
                let gpu_type = parameters.get_gpu_type();

                pretty::pretty_print_submit_streaming_response(direct_submit_read(
                    &auth_info, problem_slug, code, language, gpu_type));
            } else {
                eprintln!("Authentication failed. Please paste your Next Token into your tensara auth file.");
            }
        }
        "problems" => {
          pretty_print_problems(&parameters);
        }

        _ => unreachable!("Invalid command type"),
    }

    // Keep this code for debugging purposes, helps to see the raw response
    // let mut response_string = String::new();
    // response.read_to_string(&mut response_string).unwrap();
    // println!("{}", response_string);
}
