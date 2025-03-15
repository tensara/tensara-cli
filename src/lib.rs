pub mod client;
pub mod parser;
pub mod pretty;
use clap::ArgMatches;

pub struct Parameters {
    problem: String,
    solution_code: String,
    gpu: String,
    command_name: String,
}

impl Parameters {
    pub fn new() -> Self {
        let command_matches = match parser::parse_args(None) {
            Ok(matches) => matches,
            Err(e) => {
                pretty::print_parse_error(&e);
                std::process::exit(1);
            }
        };

        match command_matches.subcommand_name() {
            Some("checker") => {
                Self::from_subcommand("checker", parser::get_checker_matches(&command_matches))
            }
            Some("benchmark") => {
                Self::from_subcommand("benchmark", parser::get_benchmark_matches(&command_matches))
            }
            _ => {
                pretty::print_welcome_message();
                std::process::exit(0);
            }
        }
    }

    fn from_subcommand(subcommand: &str, matches: &ArgMatches) -> Self {
        let problem = parser::get_problem_name(matches).to_string();
        let solution_file = parser::get_solution_file(matches);
        let gpu = parser::get_gpu_type(matches).to_string();
        let command_name = subcommand.to_string();

        Self {
            problem,
            solution_code: Self::get_file_contents(solution_file),
            gpu,
            command_name,
        }
    }

    fn get_file_contents(solution_file: &str) -> String {
        std::fs::read_to_string(solution_file).unwrap()
    }

    pub fn get_gpu(&self) -> &String {
        &self.gpu
    }

    pub fn get_command_name(&self) -> &String {
        &self.command_name
    }

    pub fn get_problem(&self) -> &String {
        &self.problem
    }

    pub fn get_solution_code(&self) -> &String {
        &self.solution_code
    }
}
