use dotenv::dotenv;
use tensara::{
    auth::AuthInfo,
    client,
    pretty::{self, pretty_print_problems},
    Parameters,
};

const COMPILED_CHECKER_ENDPOINT: &str = env!("COMPILED_CHECKER_ENDPOINT");
const COMPILED_BENCHMARK_ENDPOINT: &str = env!("COMPILED_BENCHMARK_ENDPOINT");
const COMPILED_SUBMIT_ENDPOINT: &str = env!("COMPILED_SUBMIT_ENDPOINT");

fn main() {
    #[cfg(debug_assertions)]
    dotenv().ok();

    let auth_info = AuthInfo::load();
    let parameters: Parameters = Parameters::new(None);

    match parameters.get_command_name().as_str() {
        "checker" | "benchmark" | "submit" => {
            execute_problem_command(&parameters, &auth_info);
        }
        "problems" => {
            pretty_print_problems(&parameters);
        }
        "auth" => {
            execute_auth_command(&parameters);
        }
        _ => unreachable!("Invalid command type"),
    }

    // Keep this code for debugging purposes, helps to see the raw response
    // let mut response_string = String::new();
    // response.read_to_string(&mut response_string).unwrap();
    // println!("{}", response_string);
}

fn execute_problem_command(parameters: &Parameters, auth_info: &AuthInfo) {
    if !parameters.is_problem_command() {
        unreachable!("Invalid command type for problem execution");
    }

    let checker_endpoint =
        std::env::var("CHECKER_ENDPOINT").unwrap_or_else(|_| COMPILED_CHECKER_ENDPOINT.to_string());
    let benchmark_endpoint = std::env::var("BENCHMARK_ENDPOINT")
        .unwrap_or_else(|_| COMPILED_BENCHMARK_ENDPOINT.to_string());
    let submit_endpoint =
        std::env::var("SUBMIT_ENDPOINT").unwrap_or_else(|_| COMPILED_SUBMIT_ENDPOINT.to_string());

    let command_type = parameters.get_command_name();
    let gpu_type = parameters.get_gpu_type();
    let problem_slug = parameters.get_problem_slug();
    let language = parameters.get_language();
    let dtype = parameters.get_dtype();
    let code = parameters.get_solution_code();

    if auth_info.is_valid() {
        let response = match command_type.as_str() {
            "benchmark" => client::send_post_request_to_endpoint(
                &benchmark_endpoint,
                problem_slug,
                code,
                dtype,
                language,
                gpu_type,
                auth_info,
            ),
            "checker" => client::send_post_request_to_endpoint(
                &checker_endpoint,
                problem_slug,
                code,
                dtype,
                language,
                gpu_type,
                auth_info,
            ),
            "submit" => client::send_post_request_to_endpoint(
                &submit_endpoint,
                problem_slug,
                code,
                dtype,
                language,
                gpu_type,
                auth_info,
            ),
            _ => unreachable!("Invalid command type for problem execution"),
        };

        match command_type.as_str() {
            "benchmark" => pretty::pretty_print_benchmark_response(response),
            "checker" => pretty::pretty_print_checker_streaming_response(response),
            "submit" => pretty::pretty_print_submit_response(response),
            _ => unreachable!("Invalid command type for problem execution"),
        }
    }
}

fn execute_auth_command(parameters: &Parameters) {
    let token = parameters.get_token();
    let auth_info = AuthInfo::new(token.unwrap().to_string(), "Tensara".to_string());
    auth_info.save();
}

#[cfg(test)]
mod tests {
    use tensara::auth::AuthInfo;
    use tensara::problems::{is_valid_problem_slug};

    #[test]
    fn test_is_valid_problem_slug() {
        let slug = "vector-addition";
        assert!(is_valid_problem_slug(slug));
    }

    #[test]
    fn test_is_not_valid_problem_slug() {
        let slug = "vector-adition";
        assert!(!is_valid_problem_slug(slug));
    }

    #[test]
    fn test_auth_info() {
        let auth_info = AuthInfo::new("test_token".to_string(), "Tensara".to_string());
        auth_info.save();
        let loaded_auth_info = AuthInfo::load();
        assert_eq!(auth_info.access_token, loaded_auth_info.access_token);
    }
}
