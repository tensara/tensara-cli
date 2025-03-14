pub mod parser;
use clap::ArgMatches;
use parser::{ProblemNames, GPU};
use std::path::PathBuf;

pub struct CliInput {
    problem_name: ProblemNames,
    solution_file: PathBuf,
    gpu_type: GPU,
}

impl CliInput {
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
        let problem_name = parser::get_problem_name(matches);
        let solution_file = parser::get_solution_file(matches);
        let gpu_type = parser::get_gpu_type(matches);

        Self {
            problem_name: *problem_name,
            solution_file: PathBuf::from(solution_file),
            gpu_type: *gpu_type,
        }
    }

    pub fn display(&self) {
        println!("Problem name: {:?}", self.problem_name);
        println!("Solution file: {:?}", self.solution_file);
        println!("GPU type: {:?}", self.gpu_type);
    }
}
