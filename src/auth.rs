use serde::{Deserialize, Serialize};
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


pub fn save_token(token: String) {
    let auth_info = AuthInfo::new(token, "Tensara".to_string());
    auth_info.save();
}
