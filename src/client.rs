use std::io::{self, Read, Write};

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

pub fn send_post_request(endpoint: &str, payload: &str) {
    let client = Client::new();
    let mut response = client
        .post(endpoint)
        .body(payload.to_string())
        .send()
        .unwrap();

    let mut buffer = [0; 1024];
    let stdout = io::stdout();
    let mut handle = stdout.lock();

    while let Ok(size) = response.read(&mut buffer) {
        if size == 0 {
            break;
        }
        handle.write_all(&buffer[0..size]).unwrap();
        handle.flush().unwrap();
    }
}
