use std::io::Read;

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

    let mut response = client::send_post_request(
        endpoint,
        &parameters.get_solution_code(),
        &parameters.get_problem(),
    );

    match command_type.as_str() {
        "benchmark" => pretty::pretty_print_benchmark_response(response),
        "checker" => pretty::pretty_print_checker_streaming_response(response),
        _ => unreachable!("Invalid command type"),
    }

    // let mut response_string = String::new();
    // response.read_to_string(&mut response_string).unwrap();
    // println!("{}", response_string);
}
