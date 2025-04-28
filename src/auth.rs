use crate::trpc::{get_all_problems, Problem};
use oauth2::basic::BasicClient;
use oauth2::{
    url, AuthUrl, AuthorizationCode, ClientId, ClientSecret, CsrfToken, RedirectUrl, Scope,
    TokenUrl,
};
use reqwest::blocking::Client as ReqwestClient;
use serde::{Deserialize, Serialize};
use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use url::Url;

const AUTH_SERVER: &str = "http://localhost:8080";
const COMPILED_CLIENT_ID: &str = env!("COMPILED_CLIENT_ID");
const COMPILED_CLIENT_SECRET: &str = env!("COMPILED_CLIENT_SECRET");

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AuthInfo {
    pub access_token: String,
    pub token_type: String,
    pub expires_at_secs: Option<u64>,
    pub github_username: String,
    pub nextauth_session_token: Option<String>,
}

impl AuthInfo {
    pub fn is_expired(&self) -> bool {
        match self.expires_at_secs {
            Some(expires_secs) => {
                match SystemTime::now().duration_since(UNIX_EPOCH) {
                    Ok(now_duration) => now_duration.as_secs() > expires_secs,
                    Err(_) => false, // Error handling
                }
            }
            None => false,
        }
    }

    pub fn system_time(&self) -> Option<SystemTime> {
        self.expires_at_secs
            .map(|secs| UNIX_EPOCH + Duration::from_secs(secs))
    }
}

pub struct GitHubAuth {
    client_id: String,
    client_secret: String,
    auth_file_path: PathBuf,
}

impl GitHubAuth {
    pub fn new() -> Self {
        // Get GitHub OAuth client ID and secret from environment variables
        let client_id =
            std::env::var("CLIENT_ID").unwrap_or_else(|_| COMPILED_CLIENT_ID.to_string());
        let client_secret =
            std::env::var("CLIENT_SECRET").unwrap_or_else(|_| COMPILED_CLIENT_SECRET.to_string());

        // Create auth token directory if it doesn't exist
        let home_dir = dirs::home_dir().expect("Could not find home directory");
        let auth_dir = home_dir.join(".tensara");
        std::fs::create_dir_all(&auth_dir).expect("Failed to create auth directory");

        let auth_file_path = auth_dir.join("auth.json");

        Self {
            client_id,
            client_secret,
            auth_file_path,
        }
    }

    pub fn get_auth_info(&self) -> Option<AuthInfo> {
        // Check if we have stored auth info already
        if self.auth_file_path.exists() {
            match File::open(&self.auth_file_path) {
                Ok(file) => {
                    match serde_json::from_reader(file) {
                        Ok(auth_info) => {
                            let auth_info: AuthInfo = auth_info;
                            // If token is not expired, return it
                            if !auth_info.is_expired() {
                                return Some(auth_info);
                            }
                        }
                        Err(_) => {
                            // If we can't parse the file, we'll just authenticate again
                        }
                    }
                }
                Err(_) => {
                    // If we can't open the file, we'll just authenticate again
                }
            }
        }

        // No valid auth info found, need to authenticate
        None
    }

    pub fn authenticate(&self) -> AuthInfo {
        // Set up client ID and secret
        let github_client_id = ClientId::new(self.client_id.clone());
        let github_client_secret = ClientSecret::new(self.client_secret.clone());

        // Set up auth and token URLs
        let auth_url = AuthUrl::new("https://github.com/login/oauth/authorize".to_string())
            .expect("Invalid authorization endpoint URL");

        let token_url = TokenUrl::new("https://github.com/login/oauth/access_token".to_string())
            .expect("Invalid token endpoint URL");

        // Create the OAuth client
        let client = BasicClient::new(github_client_id)
            .set_client_secret(github_client_secret)
            .set_auth_uri(auth_url)
            .set_token_uri(token_url)
            .set_redirect_uri(
                RedirectUrl::new(format!("{}/callback", AUTH_SERVER))
                    .expect("Invalid redirect URL"),
            );

        // Generate the authorization URL
        let (authorize_url, csrf_state) = client
            .authorize_url(CsrfToken::new_random)
            .add_scope(Scope::new("read:user".to_string()))
            .url();

        // Print the URL and wait for user to authenticate
        println!("Opening browser for GitHub authentication...");
        if let Err(e) = webbrowser::open(authorize_url.as_str()) {
            println!(
                "Failed to open web browser automatically. Please open this URL manually: {}",
                authorize_url
            );
            println!("Error: {}", e);
        }

        // Start a local web server to listen for the callback
        let listener = TcpListener::bind("127.0.0.1:8080").unwrap();
        println!("Waiting for GitHub callback...");

        // Listen for the callback
        let (code, state) = {
            let Some(mut stream) = listener.incoming().flatten().next() else {
                panic!("Listener terminated without accepting a connection");
            };

            let mut reader = BufReader::new(&stream);
            let mut request_line = String::new();
            reader.read_line(&mut request_line).unwrap();

            let redirect_path = request_line.split_whitespace().nth(1).unwrap();
            let url = Url::parse(&("http://localhost".to_string() + redirect_path)).unwrap();

            let code = url
                .query_pairs()
                .find(|(key, _)| key == "code")
                .map(|(_, code)| AuthorizationCode::new(code.into_owned()))
                .unwrap();

            let state = url
                .query_pairs()
                .find(|(key, _)| key == "state")
                .map(|(_, state)| CsrfToken::new(state.into_owned()))
                .unwrap();

            // Send a response to close the browser tab
            let message =
                "Authentication successful! You can close this tab and return to the CLI.";
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {}\r\n\r\n<html><body><h1>{}</h1></body></html>",
                message.len() + 21, // +21 for the HTML tags
                message
            );
            stream.write_all(response.as_bytes()).unwrap();

            (code, state)
        };

        // Verify the CSRF token
        if state.secret() != csrf_state.secret() {
            panic!("CSRF token mismatch");
        }

        // Exchange the code directly with GitHub API using reqwest
        let client = ReqwestClient::new();
        let response = client
            .post("https://github.com/login/oauth/access_token")
            .header("Accept", "application/json")
            .form(&[
                ("client_id", self.client_id.as_str()),
                ("client_secret", self.client_secret.as_str()),
                ("code", code.secret()),
                ("redirect_uri", format!("{}/callback", AUTH_SERVER).as_str()),
            ])
            .send()
            .expect("Failed to send token request");

        if !response.status().is_success() {
            panic!("GitHub returned error: {}", response.status());
        }

        #[derive(Deserialize)]
        struct TokenResponse {
            access_token: String,
            token_type: String,
        }

        let token: TokenResponse = response.json().expect("Failed to parse token response");

        // Calculate expiration time (GitHub tokens by default expire in 8 hours)
        let expires_at = SystemTime::now() + Duration::from_secs(8 * 60 * 60);

        // Convert to seconds since UNIX epoch
        let expires_at_secs = expires_at
            .duration_since(UNIX_EPOCH)
            .ok()
            .map(|d| d.as_secs());

        // Get the GitHub username
        let github_username = self.get_github_username(&token.access_token);

        // Create the auth info
        let auth_info = AuthInfo {
            access_token: token.access_token,
            token_type: token.token_type,
            expires_at_secs,
            github_username,
            nextauth_session_token: None, // This will be set later
        };

        // Save the auth info
        self.save_auth_info(&auth_info);

        pull_problems();

        auth_info
    }

    fn get_github_username(&self, access_token: &str) -> String {
        let client = ReqwestClient::new();
        let response = client
            .get("https://api.github.com/user")
            .header("Authorization", format!("token {}", access_token))
            .header("User-Agent", "tensara-cli")
            .send()
            .expect("Failed to get user info from GitHub");

        if response.status().is_success() {
            #[derive(Deserialize)]
            struct GitHubUser {
                login: String,
            }

            response
                .json::<GitHubUser>()
                .expect("Failed to parse GitHub user info")
                .login
        } else {
            panic!("Failed to get GitHub username: {}", response.status());
        }
    }

    fn save_auth_info(&self, auth_info: &AuthInfo) {
        let file = File::create(&self.auth_file_path).expect("Failed to create auth file");
        serde_json::to_writer(file, auth_info).expect("Failed to write auth info");
    }
}

pub fn ensure_authenticated() -> AuthInfo {
    let auth = GitHubAuth::new();

    match auth.get_auth_info() {
        Some(auth_info) => auth_info,
        None => {
            let auth_info = auth.authenticate();
            auth_info
        }
    }
}

pub fn ensure_authenticated_next() -> bool {
    let auth = GitHubAuth::new();
    let auth_info = auth.get_auth_info().unwrap();
    match auth_info.nextauth_session_token {
        Some(_) => true,
        None => false,
    }
}

/*
* Do not delete while purging auth.rs
* Function to pull problems from the server and update the auth.json file
* This function is called after authentication
* We could also have this as a separate command to update problems
*/

pub fn pull_problems() {
    let auth = GitHubAuth::new();

    let auth_info = match auth.get_auth_info() {
        Some(info) => info,
        None => {
            println!("No existing auth found. Authenticating...");
            auth.authenticate()
        }
    };

    println!("Fetching problems...");
    let problems = get_all_problems().unwrap_or_else(|_| {
        eprintln!("Failed to fetch problems.");
        std::process::exit(1);
    });

    #[derive(Serialize, Deserialize, Clone, Debug)]
    struct ExtendedAuthInfo {
        #[serde(flatten)]
        auth_info: AuthInfo,
        problems: Vec<Problem>,
    }

    let extended_auth_info = ExtendedAuthInfo {
        auth_info,
        problems,
    };

    let file = File::create(auth.auth_file_path).expect("Failed to create auth file");
    serde_json::to_writer_pretty(file, &extended_auth_info).expect("Failed to write auth info");

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
