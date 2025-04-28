use crate::trpc::{get_all_problems, Problem};
use serde::{Deserialize, Serialize};
use std::fs;
use std::fs::File;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AuthInfo {
    pub access_token: String,
    pub token_type: String,
}

impl AuthInfo {
    pub fn new(access_token: String, token_type: String) -> Self {
        Self {
            access_token,
            token_type,
        }
    }

    pub fn save(&self) {
        let auth_path = dirs::home_dir()
            .expect("Could not find home directory")
            .join(".tensara")
            .join("auth.json");

        let file = File::create(&auth_path).expect("Failed to create auth file");
        serde_json::to_writer(file, &self).expect("Failed to write auth info");
    }

    pub fn load() -> Self {
        let auth_path = dirs::home_dir()
            .expect("Could not find home directory")
            .join(".tensara")
            .join("auth.json");

        let file = File::open(&auth_path).expect("Failed to open auth file");
        serde_json::from_reader(file).expect("Failed to read auth info")
    }

    pub fn is_valid(&self) -> bool {
        !self.access_token.is_empty()
    }

    pub fn write_to_auth_file(&self) {
        let auth_path = dirs::home_dir()
            .expect("Could not find home directory")
            .join(".tensara")
            .join("auth.json");

        let file = File::create(&auth_path).expect("Failed to create auth file");
        serde_json::to_writer(file, &self).expect("Failed to write auth info");
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct AuthInfoWithProblems {
    #[serde(flatten)]
    auth_info: AuthInfo,
    problems: Vec<Problem>,
}
impl AuthInfoWithProblems {
    pub fn new(auth_info: AuthInfo, problems: Vec<Problem>) -> Self {
        Self {
            auth_info,
            problems,
        }
    }

    pub fn save(&self) {
        let auth_path = dirs::home_dir()
            .expect("Could not find home directory")
            .join(".tensara")
            .join("auth.json");

        let file = File::create(&auth_path).expect("Failed to create auth file");
        serde_json::to_writer(file, &self).expect("Failed to write auth info");
    }
}

pub fn save_token(token: String) {
    let auth_info = AuthInfo::new(token, "Tensara".to_string());
    auth_info.save();
}

pub fn pull_problems() {
    let auth_info = AuthInfo::load();
    if !auth_info.is_valid() {
        eprintln!("No valid auth found. Please authenticate first.");
        std::process::exit(1);
    }

    println!("Fetching problems...");
    let problems = get_all_problems().unwrap_or_else(|_| {
        eprintln!("Failed to fetch problems.");
        std::process::exit(1);
    });

    let auth_info_with_problems = AuthInfoWithProblems::new(auth_info, problems);
    auth_info_with_problems.save();

    println!("Pulled problems and updated auth.json successfully.");
}

pub fn is_valid_problem_slug(slug: &str) -> bool {
    let auth_path = dirs::home_dir()
        .expect("Could not find home directory")
        .join(".tensara")
        .join("auth.json");

    if let Ok(auth_contents) = fs::read_to_string(auth_path) {
        if let Ok(auth_json) = serde_json::from_str::<serde_json::Value>(&auth_contents) {
            if let Some(problems) = auth_json.get("problems").and_then(|p| p.as_array()) {
                return problems.iter().any(|problem| {
                    problem
                        .get("slug")
                        .and_then(|s| s.as_str())
                        .map(|s| s.eq_ignore_ascii_case(slug))
                        .unwrap_or(false)
                });
            }
        }
    }

    false
}
