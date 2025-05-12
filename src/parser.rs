use clap::{builder::TypedValueParser, command, Arg, ArgMatches, Command};
use std::ffi::OsStr;
use std::path::Path;
use crate::problems::is_valid_problem_slug;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GPU {
    T4,
    A100,
    A100_80GB,
    H100,
    L4,
    L40s,
}

#[derive(Clone)]
struct SolutionFile;

#[derive(Clone)]
struct ProblemNameParser;

#[derive(Clone)]
struct GPUParser;

// impl std::fmt::Display for ProblemNames {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         match self {
//             Self::Conv1d => write!(f, "conv-1d"),
//             Self::Conv2d => write!(f, "conv-2d"),
//             Self::GemmRelu => write!(f, "gemm-relu"),
//             Self::LeakyRelu => write!(f, "leaky-relu"),
//             Self::MatrixMultiplication => write!(f, "matrix-multiplication"),
//             Self::MatrixVector => write!(f, "matrix-vector"),
//             Self::Relu => write!(f, "relu"),
//             Self::SquareMatmul => write!(f, "square-matmul"),
//             Self::VectorAddition => write!(f, "vector-addition"),
//         }
//     }
// }

impl std::fmt::Display for GPU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::T4 => write!(f, "T4"),
            Self::A100 => write!(f, "A100"),
            Self::A100_80GB => write!(f, "A100_80GB"),
            Self::H100 => write!(f, "H100"),
            Self::L4 => write!(f, "L4"),
            Self::L40s => write!(f, "L40s"),
        }
    }
}

impl TypedValueParser for ProblemNameParser {
    type Value = String;  // <--- now it just returns the slug string

    fn parse_ref(
        &self,
        _: &clap::Command,
        _: Option<&Arg>,
        value: &OsStr,
    ) -> Result<Self::Value, clap::Error> {
        let value_str = value.to_string_lossy().to_string();

        if is_valid_problem_slug(&value_str) {
            Ok(value_str)
        } else {
            Err(clap::Error::raw(
                clap::error::ErrorKind::InvalidValue,
                format!("Unknown problem slug: {}", value_str),
            ))
        }
    }
}

impl TypedValueParser for GPUParser {
    type Value = GPU;

    fn parse_ref(
        &self,
        _: &clap::Command,
        _: Option<&Arg>,
        value: &OsStr,
    ) -> Result<Self::Value, clap::Error> {
        let value_str = value.to_string_lossy().to_string();

        match value_str.as_str() {
            "T4" => Ok(GPU::T4),
            "A100" => Ok(GPU::A100),
            "A100_80GB" => Ok(GPU::A100_80GB),
            "H100" => Ok(GPU::H100),
            "L4" => Ok(GPU::L4),
            "L40s" => Ok(GPU::L40s),
            _ => Err(clap::Error::raw(
                clap::error::ErrorKind::InvalidValue,
                format!("GPU_TYPE: {}", value_str),
            )),
        }
    }
}

impl TypedValueParser for SolutionFile {
    type Value = String;

    fn parse_ref(
        &self,
        _: &clap::Command,
        _: Option<&Arg>,
        value: &OsStr,
    ) -> Result<Self::Value, clap::Error> {
        let path_str = value.to_string_lossy().to_string();
        let path = Path::new(&path_str);

        if !path.exists() {
            return Err(clap::Error::raw(
                clap::error::ErrorKind::InvalidValue,
                format!("SOLUTION_FILE: {}", path_str),
            ));
        }
        if !path.is_file() {
            return Err(clap::Error::raw(
                clap::error::ErrorKind::InvalidValue,
                format!("SOLUTION_FILE: {}", path_str),
            ));
        }

        if let Some(ext) = path.extension() {
            let ext_str = ext.to_string_lossy().to_lowercase();
            if ext_str != "cu" && ext_str != "py" {
                return Err(clap::Error::raw(
                    clap::error::ErrorKind::InvalidValue,
                    format!("SOLUTION_FILE: {}", path_str),
                ));
            } 
        } else {
            return Err(clap::Error::raw(
                clap::error::ErrorKind::InvalidValue,
                format!("SOLUTION_FILE: {}", path_str),
            ));
        }

        Ok(path_str)
    }
}

pub fn parse_args(args: Option<Vec<&str>>) -> Result<ArgMatches, clap::Error> {
    let command = command!()
            .about(
                "A CLI tool for submitting and benchmarking solutions to GPU programming problems on tensara. \
                \nFind available problems at https://tensara.org/problems",
            )
            .subcommand(
                Command::new("submit")
                    .about("Submit a solution to a problem with your tensara account")
                    .arg_required_else_help(true)
                    .arg(
                        Arg::new("gpu_type")
                            .short('g')
                            .value_name("GPU_TYPE")
                            .help("Type of the GPU to use")
                            .default_value("T4")
                            .required(false)
                            .value_parser(GPUParser),
                    )
                    .arg(
                        Arg::new("problem_name")
                            .short('p')
                            .long("problem")
                            .value_name("PROBLEM_NAME")
                            .value_parser(ProblemNameParser)
                            .help("Name of the problem to test")
                            .required(true),
                    )
                    .arg(
                        Arg::new("solution_file")
                            .short('s')
                            .long("solution")
                            .value_name("SOLUTION_FILE")
                            .help("Relative path to the solution file")
                            .value_parser(SolutionFile)
                            .required(true),
                    )                    
            )
            .subcommand(
                Command::new("checker")
                    .about("Check if your solution is correct")
                    .arg_required_else_help(true)
                    .arg(
                        Arg::new("gpu_type")
                            .short('g')
                            .value_name("GPU_TYPE")
                            .help("Type of the GPU to use")
                            .default_value("T4")
                            .required(false)
                            .value_parser(GPUParser),
                    )
                    .arg(
                        Arg::new("problem_name")
                            .short('p')
                            .long("problem")
                            .value_name("PROBLEM_NAME")
                            .value_parser(ProblemNameParser)
                            .help("Name of the problem to test")
                            .required(true),
                    )
                    .arg(
                        Arg::new("solution_file")
                            .short('s')
                            .long("solution")
                            .value_name("SOLUTION_FILE")
                            .help("Relative path to the solution file")
                            .value_parser(SolutionFile)
                            .required(true),
                    )                    
            )
            .subcommand(
                Command::new("benchmark")
                    .about("Benchmark a solution and get the performance metrics for a given problem")
                    .arg_required_else_help(true)
                    .arg(
                        Arg::new("gpu_type")
                            .short('g')
                            .value_name("GPU_TYPE")
                            .help("Type of the GPU to use")
                            .default_value("T4")
                            .required(false)
                            .value_parser(GPUParser),
                    )
                    .arg(
                        Arg::new("problem_name")
                            .short('p')
                            .long("problem")
                            .value_name("PROBLEM_NAME")
                            .help("Name of the problem to test")
                            .required(true)
                            .value_parser(ProblemNameParser),
                    )
                    .arg(
                        Arg::new("solution_file")
                            .short('s')
                            .long("solution")
                            .value_name("SOLUTION_FILE")
                            .help("Relative path to the solution file")
                            .value_parser(SolutionFile)
                            .required(true)
                    )
            )
            .subcommand(
                Command::new("problems")
                    .about("List all problems")
                    .arg(
                        Arg::new("field")
                            .short('f')
                            .long("field")
                            .value_name("FIELD")
                            .help("Field(s) to display: slug, title, difficulty, author, tags")
                            .action(clap::ArgAction::Append)
                            .required(false),
                    )
                    .arg(
                        Arg::new("sort_by")
                            .short('s')
                            .long("sort-by")
                            .value_name("FIELD")
                            .help("Field to sort by: slug, title, difficulty, author")
                            .required(false),
                    )
            )
            .subcommand(
                Command::new("auth")
                    .about("Authenticate with the Tensara API to submit solutions")
                    .arg(
                        Arg::new("token")
                            .short('t')
                            .long("token")
                            .value_name("TOKEN")
                            .help("Provide authentication token directly")
                            .required(false),
                    ),
            );

    if let Some(args) = args {
        command.try_get_matches_from(args)
    } else {
        command.try_get_matches()
    }
}

pub fn get_problem_name(matches: &ArgMatches) -> &String {
    matches.get_one::<String>("problem_name").unwrap()
}

pub fn get_solution_file(matches: &ArgMatches) -> &String {
    matches.get_one::<String>("solution_file").unwrap()
}

pub fn get_gpu_type(matches: &ArgMatches) -> &GPU {
    matches.get_one::<GPU>("gpu_type").unwrap()
}

pub fn get_checker_matches(matches: &ArgMatches) -> &ArgMatches {
    matches.subcommand_matches("checker").unwrap()
}

pub fn get_benchmark_matches(matches: &ArgMatches) -> &ArgMatches {
    matches.subcommand_matches("benchmark").unwrap()
}

pub fn get_auth_matches(matches: &ArgMatches) -> &ArgMatches {
    matches.subcommand_matches("auth").unwrap()
}

pub fn get_submit_matches(matches: &ArgMatches) -> &ArgMatches {
    matches.subcommand_matches("submit").unwrap()
}
pub fn get_problems_matches(matches: &ArgMatches) -> &ArgMatches {
    matches.subcommand_matches("problems").unwrap()
}

pub fn get_language_type(matches: &ArgMatches) -> &String {
    matches.get_one::<String>("language").unwrap()
}



