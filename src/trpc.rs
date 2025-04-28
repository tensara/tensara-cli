/*
* Use these functions to call the tRPC endpoints from the Tensara API.
*/
use crate::auth::AuthInfo;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Problem {
    pub id: String,
    pub slug: String,
    pub title: String,
    pub difficulty: Option<String>,
    pub author: Option<String>,
    pub tags: Option<Vec<String>>,
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

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateSubmissionInput {
    pub problem_slug: String,
    pub code: String,
    pub language: String,
    pub gpu_type: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Submission {
    pub id: String,
    pub status: Option<String>,
    pub language: Option<String>,
    pub gpu_type: Option<String>,
    pub problem_id: String,
    pub user_id: String,
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

/*
* Use this function to get all problems
*/
pub fn get_all_problems() -> Result<Vec<Problem>, Box<dyn std::error::Error>> {
    let client = Client::new();
    let url = "https://tensara.org/api/trpc/problems.getAll";

    let response = client.get(url).header("User-Agent", "tensara-cli").send()?;

    let parsed: TrpcResult<Vec<Problem>> = response.json()?;
    Ok(parsed.result.data.json)
}

/*
* Function to demonstrate how to call the tRPC endpoint for user stats
*/
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

/*
* Use this function to get the problem details by slug
*/

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

pub fn direct_submit_read(auth: &AuthInfo, submission_id: &str) -> reqwest::blocking::Response {
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

// pub fn direct_submit_read(auth: &AuthInfo, submission_id: &str) -> impl Read {
//     let client = Client::new();
//     let url = "https://tensara.org/api/submissions/direct-submit";

//     let body = serde_json::json!({ "submissionId": submission_id });

//     client
//         .post(url)
//         .header("Content-Type", "application/json")
//         .header("User-Agent", "tensara-cli")
//         .header(
//             "Cookie",
//             format!(
//                 "__Secure-next-auth.session-token={}",
//                 auth.nextauth_session_token.as_ref().unwrap()
//             ),
//         )
//         .json(&body)
//         .send()
//         .expect("Failed to send request")
// }
