use crate::auth::AuthInfo;
use reqwest::blocking::Client;
use serde::Deserialize;
use serde_json::Value;
use urlencoding::encode;

#[derive(Debug, Deserialize)]
pub struct Problem {
    pub id: String,
    pub slug: String,
    pub title: String,
    pub difficulty: Option<String>,
    pub author: Option<String>,
    pub tags: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct TrpcResult<T> {
    result: TrpcData<T>,
}

#[derive(Debug, Deserialize)]
struct TrpcData<T> {
    data: TrpcJson<T>,
}

#[derive(Debug, Deserialize)]
struct TrpcJson<T> {
    json: T,
}

pub fn get_all_problems() -> Result<Vec<Problem>, Box<dyn std::error::Error>> {
    let client = Client::new();
    let url = "https://tensara.org/api/trpc/problems.getAll";

    let response = client.get(url).header("User-Agent", "tensara-cli").send()?;

    let parsed: TrpcResult<Vec<Problem>> = response.json()?;
    Ok(parsed.result.data.json)
}

pub fn call_trpc_user_stats(auth: &AuthInfo) {
    let session_cookie = format!(
        "__Secure-next-auth.session-token={}",
        auth.nextauth_session_token
            .as_ref()
            .expect("No session token found")
    );

    let client = Client::new();
    let url = "https://tensara.org/api/trpc/problems.getUserStats";

    let response = client
        .get(url)
        .header("Cookie", session_cookie)
        .header("User-Agent", "tensara-cli")
        .send()
        .expect("Failed to send request");

    let text = response.text().unwrap();
    println!("tRPC Response: {}", text);
}

#[derive(Debug, Deserialize)]
pub struct ProblemDetails {
    pub id: String,
    pub slug: String,
    pub title: String,
    pub difficulty: Option<String>,
    pub author: Option<String>,
    pub tags: Option<Vec<String>>,
    pub description: Option<String>,
}

pub fn get_problem_by_slug(slug: &str) -> Result<ProblemDetails, Box<dyn std::error::Error>> {
    let client = Client::new();
    let input_json = serde_json::json!({ "json": { "slug": slug } }).to_string();
    let encoded_input = urlencoding::encode(&input_json).into_owned();
    let url = format!(
        "https://tensara.org/api/trpc/problems.getById?input={}",
        encoded_input
    );

    let response = client
        .get(&url)
        .header("User-Agent", "tensara-cli")
        .send()?;

    let parsed: TrpcResult<ProblemDetails> = response.json()?;
    Ok(parsed.result.data.json)
}


