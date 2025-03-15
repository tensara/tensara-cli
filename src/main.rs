use std::io::Read;

use dotenv::dotenv;
use tensara::{client, pretty, Payload};

fn main() {
    dotenv().ok();

    let binding = std::env::var("MODAL_SLUG").unwrap();
    let endpoint = format!("{}/benchmark-T4", binding);
    let endpoint = endpoint.as_str();

    let payload = Payload::new();
    let payload_str = serde_json::to_string(&payload).unwrap();
    let mut response = client::send_post_request(endpoint, &payload_str);
    pretty::pretty_print_benchmark_response(response);
    // let mut response_str = String::new();
    // response.read_to_string(&mut response_str).unwrap();
    // println!("{}", response_str);
}
