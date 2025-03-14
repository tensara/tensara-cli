use clap::{command, Arg, ArgMatches, Command};

pub fn get_matches() -> ArgMatches {
    command!()
            .about(
                "A CLI tool for submitting and benchmarking solutions to GPU programming problems on tensara. \
                \nFind available problems at https://tensara.org/problems",
            )
            .subcommand_required(true)
            .subcommand(
                Command::new("checker")
                    .about("Submit a solution to a problem and check if it is correct")
                    .arg(
                        Arg::new("problem_name")
                            .short('p')
                            .long("problem")
                            .value_name("PROBLEM_NAME")
                            .help("Name of the problem to test")
                            .required(true),
                    )
                    .arg(
                        Arg::new("solution_file")
                            .short('s')
                            .long("solution")
                            .value_name("SOLUTION_FILE")
                            .help("Relative path to the solution file")
                            .required(true),
                    )
                    .arg(
                        Arg::new("gpu_type")
                            .short('g')
                            .long("gpu")
                            .value_name("GPU_TYPE")
                            .help("Type of the GPU to use")
                            .default_value("T4")
                            .required(false),
                    )
            )
            .subcommand(
                Command::new("benchmark")
                    .about("Benchmark a solution and get the performance metrics for a given problem")
                    .arg(
                        Arg::new("problem_name")
                            .short('p')
                            .long("problem")
                            .value_name("PROBLEM_NAME")
                            .help("Name of the problem to test")
                            .required(true),
                    )
                    .arg(
                        Arg::new("solution_file")
                            .short('s')
                            .long("solution")
                            .value_name("SOLUTION_FILE")
                            .help("Relative path to the solution file")
                            .required(true),
                    )
                    .arg(
                        Arg::new("gpu_type")
                            .short('g')
                            .long("gpu")
                            .value_name("GPU_TYPE")
                            .help("Type of the GPU to use")
                            .default_value("T4")
                            .required(false),
                    )
            )
            .get_matches()
}
