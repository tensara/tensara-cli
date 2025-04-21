use crate::auth::AuthInfo;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::Read;
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

#[derive(Debug, Deserialize, Serialize)]
pub struct CreateSubmissionInput {
    pub problemSlug: String,
    pub code: String,
    pub language: String,
    pub gpuType: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Submission {
    pub id: String,
    pub status: Option<String>,
    pub language: Option<String>,
    pub gpuType: Option<String>,
    pub problemId: String,
    pub userId: String,
}

pub fn create_submission(
    auth: &AuthInfo,
    problem_slug: &str,
    code: &str,
    language: &str,
    gpu_type: &str,
) -> Result<Submission, Box<dyn std::error::Error>> {
    let client = Client::new();
    let url = "https://tensara.org/api/trpc/problems.createSubmission";

    let input_json = serde_json::json!({

            "json": {
                "problemSlug": problem_slug,
                "code": code,
                "language": language,
                "gpuType": gpu_type
            }
    });

    let response = client
        .post(url)
        .header("Content-Type", "application/json")
        .header("User-Agent", "tensara-cli")
        .header(
            "Cookie",
            format!(
                "__Secure-next-auth.session-token={}",
                auth.nextauth_session_token.as_ref().unwrap()
            ),
        )
        .json(&input_json)
        .send()?;

    if !response.status().is_success() {
        let error_text = response.text()?;
        return Err(format!("Error from server: {}", error_text).into());
    }

    let parsed: TrpcResult<Submission> = response.json()?;
    Ok(parsed.result.data.json)
}

pub fn direct_submit(
    auth: &AuthInfo,
    submission_id: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new();
    let url = "https://tensara.org/api/submissions/direct-submit";

    let body = serde_json::json!({ "submissionId": submission_id });

    let response = client
        .post(url)
        .header("Content-Type", "application/json")
        .header("User-Agent", "tensara-cli")
        .header(
            "Cookie",
            format!(
                "__Secure-next-auth.session-token={}",
                auth.nextauth_session_token.as_ref().unwrap()
            ),
        )
        .json(&body)
        .send()?;

    if response.status().is_success() {
        Ok(())
    } else {
        let error_text = response.text()?;
        Err(format!("Direct submit failed: {}", error_text).into())
    }
}

pub fn direct_submit_read(auth: &AuthInfo, submission_id: &str) -> impl Read {
    let client = Client::new();
    let url = "https://tensara.org/api/submissions/direct-submit";

    let body = serde_json::json!({ "submissionId": submission_id });

    client
        .post(url)
        .header("Content-Type", "application/json")
        .header("User-Agent", "tensara-cli")
        .header(
            "Cookie",
            format!(
                "__Secure-next-auth.session-token={}",
                auth.nextauth_session_token.as_ref().unwrap()
            ),
        )
        .json(&body)
        .send()
        .expect("Failed to send request")
}
