pub mod parser;

use clap::ArgMatches;

pub struct Payload {
    problem_name: String,
    solution_code: String,
    gpu_type: String,
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
        let problem_name = parser::get_problem_name(matches).to_string();
        let solution_file = parser::get_solution_file(matches);
        let gpu_type = parser::get_gpu_type(matches).to_string();

        Self {
            problem_name,
            solution_code: Self::get_file_contents(solution_file),
            gpu_type,
        }
    }

    pub fn display(&self) {
        println!("Problem name: {:?}", self.problem_name);
        println!("Solution code: {:?}", self.solution_code);
        println!("GPU type: {:?}", self.gpu_type);
    }

    fn get_file_contents(solution_file: &str) -> String {
        std::fs::read_to_string(solution_file).unwrap()
    }
}
