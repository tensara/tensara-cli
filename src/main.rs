use dotenv::dotenv;
use tensara::{client, Payload};

fn main() {
    dotenv().ok();

    let binding = std::env::var("MODAL_SLUG").unwrap();
    let endpoint = format!("{}/checker-T4", binding);
    let endpoint = endpoint.as_str();

    let payload = Payload::new();
    let payload_str = serde_json::to_string(&payload).unwrap();
    client::send_post_request(endpoint, &payload_str);
}
