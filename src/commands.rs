use clap::{command, Arg, ArgMatches};
use std::collections::HashMap;

pub fn new() -> HashMap<String, ArgMatches> {
    let mut map = HashMap::new();
    init_commands(&mut map);
    map
}

fn init_commands(map: &mut HashMap<String, ArgMatches>) {
    map.insert(
        "checker".to_string(),
        command!()
            .arg(
                Arg::new("runner_type")
                    .value_name("RUNNER_TYPE")
                    .help("Type of the runner to use: checker or benchmark")
                    .required(true),
            )
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
                    .help("Type of the GPU to use (default: T4)")
                    .required(false),
            )
            .get_matches(),
    );
}
