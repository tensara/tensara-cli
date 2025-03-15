use dotenv::dotenv;
use tensara::{client, pretty, Parameters};

fn main() {
    dotenv().ok();

    let parameters = Parameters::new();
    let command_type = parameters.get_command_name();
    let gpu = parameters.get_gpu();

    let binding = std::env::var("MODAL_SLUG").unwrap();
    let endpoint = format!("{}/{}-{}", binding, command_type, gpu);
    let endpoint = endpoint.as_str();

    let response = client::send_post_request(
        endpoint,
        &parameters.get_solution_code(),
        &parameters.get_problem(),
    );

    match command_type.as_str() {
        "benchmark" => pretty::pretty_print_benchmark_response(response),
        "checker" => pretty::pretty_print_checker_streaming_response(response),
        _ => unreachable!("Invalid command type"),
    }
}
