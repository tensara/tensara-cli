use std::io::Read;

use reqwest::blocking::Client;

pub fn send_get_request(endpoint: &str) {
    let client = Client::new();
    let response = client.get(endpoint).send().unwrap();
    if response.status().is_success() {
        println!("{}", response.text().unwrap());
    } else {
        println!("Error: {}", response.status());
    }
}

pub fn send_post_request(endpoint: &str, payload: &str) -> impl Read {
    let client = Client::new();
    client
        .post(endpoint)
        .body(payload.to_string())
        .send()
        .unwrap()
}
