pub mod client;
pub mod parser;
pub mod pretty;

use clap::ArgMatches;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Payload {
    problem: String,
    solution_code: String,
    gpu: String,
}

impl Payload {
    pub fn new() -> Self {
        let command_matches = parser::get_matches();
        match command_matches.subcommand_name() {
            Some("checker") => Self::from_subcommand(parser::get_checker_matches(&command_matches)),
            Some("benchmark") => {
                Self::from_subcommand(parser::get_benchmark_matches(&command_matches))
            }
            _ => unreachable!("Subcommand is required by clap"),
        }
    }

    fn from_subcommand(matches: &ArgMatches) -> Self {
        let problem = parser::get_problem_name(matches).to_string();
        let solution_file = parser::get_solution_file(matches);
        let gpu = parser::get_gpu_type(matches).to_string();

        Self {
            problem,
            solution_code: Self::get_file_contents(solution_file),
            gpu,
        }
    }

    fn get_file_contents(solution_file: &str) -> String {
        std::fs::read_to_string(solution_file).unwrap()
    }
}
