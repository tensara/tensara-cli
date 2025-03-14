use clap::{builder::TypedValueParser, command, value_parser, Arg, ArgMatches, Command};
use std::ffi::OsStr;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProblemNames {
    Conv1d,
    Conv2d,
    GemmRelu,
    LeakyRelu,
    MatrixMultiplication,
    MatrixVector,
    Relu,
    SquareMatmul,
    VectorAddition,
}

impl std::str::FromStr for ProblemNames {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "conv-1d" | "conv1d" => Ok(Self::Conv1d),
            "conv-2d" | "conv2d" => Ok(Self::Conv2d),
            "gemm-relu" | "gemmrelu" => Ok(Self::GemmRelu),
            "leaky-relu" | "leakyrelu" => Ok(Self::LeakyRelu),
            "matrix-multiplication" | "matrixmultiplication" => Ok(Self::MatrixMultiplication),
            "matrix-vector" | "matrixvector" => Ok(Self::MatrixVector),
            "relu" => Ok(Self::Relu),
            "square-matmul" | "squarematmul" => Ok(Self::SquareMatmul),
            "vector-addition" | "vectoraddition" => Ok(Self::VectorAddition),
            _ => Err(format!("Unknown problem name: {}", s)),
        }
    }
}

impl std::fmt::Display for ProblemNames {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Conv1d => write!(f, "conv-1d"),
            Self::Conv2d => write!(f, "conv-2d"),
            Self::GemmRelu => write!(f, "gemm-relu"),
            Self::LeakyRelu => write!(f, "leaky-relu"),
            Self::MatrixMultiplication => write!(f, "matrix-multiplication"),
            Self::MatrixVector => write!(f, "matrix-vector"),
            Self::Relu => write!(f, "relu"),
            Self::SquareMatmul => write!(f, "square-matmul"),
            Self::VectorAddition => write!(f, "vector-addition"),
        }
    }
}

// A custom parser that validates file existence and extension
#[derive(Clone)]
struct SolutionFile;

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
                format!("File does not exist: {} \n", path_str),
            ));
        }
        if !path.is_file() {
            return Err(clap::Error::raw(
                clap::error::ErrorKind::InvalidValue,
                format!("Path is not a file: {} \n", path_str),
            ));
        }

        if let Some(ext) = path.extension() {
            let ext_str = ext.to_string_lossy().to_lowercase();
            if ext_str != "cu" && ext_str != "py" {
                return Err(clap::Error::raw(
                    clap::error::ErrorKind::InvalidValue,
                    format!("File must be a .cu or .py file: {} \n", path_str),
                ));
            }
        } else {
            return Err(clap::Error::raw(
                clap::error::ErrorKind::InvalidValue,
                format!("File has no extension: {} \n", path_str),
            ));
        }

        Ok(path_str)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GPU {
    T4,
    A100,
    A100_80GB,
    H100,
    L4,
    L40s,
}

impl std::str::FromStr for GPU {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "T4" => Ok(Self::T4),
            "A100" => Ok(Self::A100),
            "A100_80GB" => Ok(Self::A100_80GB),
            "H100" => Ok(Self::H100),
            "L4" => Ok(Self::L4),
            "L40s" => Ok(Self::L40s),
            _ => Err(format!("Unknown GPU type: {}", s)),
        }
    }
}

impl std::fmt::Display for GPU {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

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
                            .value_parser(value_parser!(ProblemNames))
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
                    .arg(
                        Arg::new("gpu_type")
                            .short('g')
                            .long("gpu")
                            .value_name("GPU_TYPE")
                            .help("Type of the GPU to use")
                            .default_value("T4")
                            .required(false)
                            .value_parser(value_parser!(GPU)),
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
                            .required(true)
                            .value_parser(value_parser!(ProblemNames)),
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
                    .arg(
                        Arg::new("gpu_type")
                            .short('g')
                            .long("gpu")
                            .value_name("GPU_TYPE")
                            .help("Type of the GPU to use")
                            .default_value("T4")
                            .value_parser(value_parser!(GPU))
                            .required(false),
                    )
            )
            .get_matches()
}

pub fn get_problem_name(matches: &ArgMatches) -> &ProblemNames {
    matches.get_one::<ProblemNames>("problem_name").unwrap()
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
