pub mod parser;

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

        let mut cli_input = CliInput {
            problem_name: ProblemNames::Conv1d,
            solution_file: PathBuf::from(""),
            gpu_type: GPU::T4,
        };

        if let Some(subcommand) = command_matches.subcommand_name() {
            match subcommand {
                "checker" => {
                    let checker_matches = parser::get_checker_matches(&command_matches);
                    let problem_name = parser::get_problem_name(checker_matches);
                    let solution_file = parser::get_solution_file(checker_matches);
                    let gpu_type = parser::get_gpu_type(checker_matches);
                    cli_input = CliInput {
                        problem_name: *problem_name,
                        solution_file: PathBuf::from(solution_file),
                        gpu_type: *gpu_type,
                    };
                }
                "benchmark" => {
                    let benchmark_matches = parser::get_benchmark_matches(&command_matches);
                    let problem_name = parser::get_problem_name(benchmark_matches);
                    let solution_file = parser::get_solution_file(benchmark_matches);
                    let gpu_type = parser::get_gpu_type(benchmark_matches);
                    cli_input = CliInput {
                        problem_name: *problem_name,
                        solution_file: PathBuf::from(solution_file),
                        gpu_type: *gpu_type,
                    };
                }
                _ => unreachable!("Subcommand is required by clap"),
            }
        }
        cli_input
    }

    pub fn display(&self) {
        println!("Problem name: {:?}", self.problem_name);
        println!("Solution file: {:?}", self.solution_file);
        println!("GPU type: {:?}", self.gpu_type);
    }
}
