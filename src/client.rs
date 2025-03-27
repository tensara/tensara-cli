use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::io::Read;

#[derive(Serialize, Deserialize)]
struct Request {
    solution_code: String,
    problem: String,
    problem_def: String,
    dtype: String,
    language: String,
}

impl Request {
    pub fn new(
        solution_code: String,
        problem: String,
        problem_def: String,
        dtype: String,
        language: String,
    ) -> Self {
        Self {
            solution_code,
            problem,
            problem_def,
            dtype,
            language,
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

pub fn send_post_request(
    endpoint: &str,
    solution_code: &str,
    problem: &str,
    problem_def: &str,
    dtype: &str,
    language: &str,
) -> impl Read {
    let request = Request::new(
        solution_code.to_string(),
        problem.to_string(),
        problem_def.to_string(),
        dtype.to_string(),
        language.to_string(),
    );
    let request_json = serde_json::to_string(&request).unwrap();
    let client = Client::new();
    client
        .post(endpoint)
        .header("Content-Type", "application/json")
        .body(request_json)
        .send()
        .expect("Failed to send request")
}

pub fn get_problem_definition(endpoint: &str) -> String {
    let client = Client::new();
    let response = client.get(endpoint).send().expect("Failed to send request");
    response.text().expect("Failed to read response")
}
