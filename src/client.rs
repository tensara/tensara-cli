use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::io::Read;

use crate::auth::AuthInfo;

#[allow(non_snake_case)]
#[derive(Serialize, Deserialize)]
struct Request {
    problemSlug: String,
    code: String,
    dtype: String,
    language: String,
    gpuType: String,
}

#[allow(non_snake_case)]
impl Request {
    pub fn new(
        problemSlug: String,
        code: String,
        dtype: String,
        language: String,
        gpuType: String,
    ) -> Self {
        Self {
            problemSlug,
            code,
            dtype,
            language,
            gpuType,
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

pub fn send_post_request_to_endpoint(
    endpoint: &str,
    problem_slug: &str,
    code: &str,
    dtype: &str,
    language: &str,
    gpu_type: &str,
    auth: &AuthInfo,
) -> impl Read {
    let request = Request::new(
        problem_slug.to_string(),
        code.to_string(),
        dtype.to_string(),
        language.to_string(),
        gpu_type.to_string(),
    );
    let request_json = serde_json::to_string(&request).unwrap();
    let client = Client::new();
    client
        .post(endpoint)
        .header("Content-Type", "application/json")
        .header("User-Agent", "tensara-cli")
        .header("Authorization", format!("Bearer {}", auth.access_token))
        .body(request_json)
        .send()
        .expect("Failed to send request")
}
