pub mod auth;
pub mod client;
pub mod parser;
pub mod pretty;
pub mod trpc;
use clap::ArgMatches;

const COMPILED_PROBLEM_ENDPOINT: &str = env!("COMPILED_PROBLEM_URL");

pub struct Parameters {
    problem_def: String,
    problem: String,
    solution_code: String,
    dtype: String,
    language: String,
    command_name: String,
    gpu_type: String,
}

impl Parameters {
    pub fn new(username: Option<String>) -> Self {
        let command_matches = match parser::parse_args(None) {
            Ok(matches) => matches,
            Err(e) => match e.kind() {
                clap::error::ErrorKind::DisplayHelp => {
                    println!("{}", e.to_string());
                    std::process::exit(0);
                }
                clap::error::ErrorKind::DisplayVersion => {
                    println!("{}", e.to_string());
                    std::process::exit(0);
                }
                _ => {
                    pretty::print_parse_error(&e);
                    std::process::exit(1);
                }
            },
        };

        match command_matches.subcommand_name() {
            Some("checker") => {
                Self::from_subcommand("checker", parser::get_checker_matches(&command_matches))
            }
            Some("benchmark") => {
                Self::from_subcommand("benchmark", parser::get_benchmark_matches(&command_matches))
            }
            Some("submit") => {
                // Self::from_subcommand("submit", parser::get_submit_matches(&command_matches))
                // do nothing
                Parameters {
                    problem_def: "".to_string(),
                    problem: "".to_string(),
                    solution_code: "".to_string(),
                    dtype: "".to_string(),
                    language: "".to_string(),
                    command_name: "submit".to_string(),
                    gpu_type: "".to_string(),
                }
            }
            _ => {
                pretty::print_welcome_message(username);
                std::process::exit(0);
            }
        }
    }

    fn from_subcommand(subcommand: &str, matches: &ArgMatches) -> Self {
        let problem = parser::get_problem_name(matches).to_string();
        let solution_file = parser::get_solution_file(matches);
        let dtype = "float32".to_string();
        let gpu_type = parser::get_gpu_type(matches).to_string();
        let solution_file_extension = solution_file.split('.').last().unwrap();
        let language = match solution_file_extension {
            "py" => "python".to_string(),
            "cu" => "cuda".to_string(),
            _ => "unknown".to_string(),
        };

        let command_name = subcommand.to_string();

        let problem_endpoint = std::env::var("PROBLEM_ENDPOINT")
            .unwrap_or_else(|_| COMPILED_PROBLEM_ENDPOINT.to_string());
        let problem_endpoint = format!("{}/{}/def.py", problem_endpoint, problem);
        let problem_def = client::get_problem_definition(&problem_endpoint);

        Self {
            problem_def,
            problem,
            solution_code: Self::get_file_contents(solution_file),
            dtype,
            language: language.to_string(),
            command_name,
            gpu_type,
        }
    }

    fn get_file_contents(solution_file: &str) -> String {
        std::fs::read_to_string(solution_file).unwrap()
    }

    pub fn get_command_name(&self) -> &String {
        &self.command_name
    }

    pub fn get_problem_def(&self) -> &String {
        &self.problem_def
    }

    pub fn get_solution_code(&self) -> &String {
        &self.solution_code
    }
    pub fn get_gpu_type(&self) -> &String {
        &self.gpu_type
    }

    pub fn get_dtype(&self) -> &String {
        &self.dtype
    }

    pub fn get_language(&self) -> &String {
        &self.language
    }

    pub fn get_problem(&self) -> &String {
        &self.problem
    }
}
