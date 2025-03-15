use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::io::Read;

#[derive(Serialize, Deserialize)]
struct Request {
    solution_code: String,
    problem: String,
}

impl Request {
    pub fn new(solution_code: String, problem: String) -> Self {
        Self {
            solution_code,
            problem,
        }
    }
}

pub fn send_get_request(endpoint: &str) {
    let client = Client::new();
    match client.get(endpoint).send() {
        Ok(response) => {
            if response.status().is_success() {
                match response.text() {
                    Ok(text) => println!("{}", text),
                    Err(e) => println!("Error reading response: {}", e),
                }
            } else {
                println!("Error: {}", response.status());
            }
        }
        Err(e) => println!("Request failed: {}", e),
    }
}

pub fn send_post_request(endpoint: &str, solution_code: &str, problem: &str) -> impl Read {
    let request = Request::new(solution_code.to_string(), problem.to_string());
    let request_json = serde_json::to_string(&request).unwrap();
    let client = Client::new();
    client
        .post(endpoint)
        .header("Content-Type", "application/json")
        .body(request_json)
        .send()
        .expect("Failed to send request")
}
