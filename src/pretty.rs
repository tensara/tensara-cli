use console::style;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use serde_json::Value;
use std::io::Read;
use std::time::Duration;

pub fn pretty_print_checker_streaming_response(mut response: impl Read) {
    let multi_progress = MultiProgress::new();

    let spinner_style = ProgressStyle::default_spinner()
        .template("{spinner:.green} {prefix:.bold.dim} {wide_msg}")
        .unwrap()
        .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏");

    let progress_style = ProgressStyle::default_bar()
        .template("{prefix:.bold.white} [{bar:40.green/blue}] {pos}/{len} {msg}")
        .unwrap()
        .progress_chars("█▓▒░  ");

    let mut compilation_pb = multi_progress.add(ProgressBar::new_spinner());
    compilation_pb.set_style(spinner_style.clone());
    compilation_pb.set_prefix("🔧");
    compilation_pb.enable_steady_tick(Duration::from_millis(80));

    let mut _total_tests = 0;
    let mut completed_tests = 0;
    let mut test_progress: Option<ProgressBar> = None;
    let mut test_results: Vec<Value> = Vec::new();
    let mut buffer = [0; 1024];
    let mut data_buffer = String::new();

    while let Ok(size) = response.read(&mut buffer) {
        if size == 0 {
            break;
        }

        let chunk = String::from_utf8_lossy(&buffer[0..size]);
        data_buffer.push_str(&chunk);

        while let Some(pos) = data_buffer.find('\n') {
            let line = data_buffer[..pos].trim().to_string();
            let remaining = data_buffer[pos + 1..].to_string();
            data_buffer = remaining;

            if line.starts_with("data: ") {
                let json_str = &line["data: ".len()..];
                if let Ok(json) = serde_json::from_str::<Value>(json_str) {
                    match json["status"].as_str() {
                        Some("compiling") => {
                            compilation_pb.set_message("Compiling your code...".to_string());
                        }
                        Some("error") => {
                            compilation_pb.finish_with_message("Error detected!".to_string());
                            compilation_pb.set_prefix("❌");

                            compilation_pb.finish_and_clear();
                            multi_progress.clear().unwrap();

                            println!("\n{}", style("⚠️ ERROR OCCURRED ⚠️").red().bold());
                            println!("{}", style("═".repeat(50)).dim());

                            let error_message = json["error"].as_str().unwrap_or("Unknown error");
                            println!("{}: {}", style("Error").red().bold(), error_message);

                            if let Some(details) = json["details"].as_str() {
                                println!("\n{}", style("Details:").yellow().bold());
                                println!("{}", style("─".repeat(50)).dim());

                                let formatted_details = details
                                    .lines()
                                    .map(|line| {
                                        if line.contains("error:") {
                                            format!("{}", style(line).red())
                                        } else if line.contains("warning:") {
                                            format!("{}", style(line).yellow())
                                        } else if line.contains("^") {
                                            format!("{}", style(line).cyan())
                                        } else {
                                            line.to_string()
                                        }
                                    })
                                    .collect::<Vec<String>>()
                                    .join("\n");

                                println!("{}", formatted_details);
                            }

                            println!("{}", style("═".repeat(50)).dim());
                            println!(
                                "{}",
                                style("Please fix the errors and try again.")
                                    .yellow()
                                    .bold()
                            );

                            return;
                        }
                        Some("running") => {
                            compilation_pb
                                .finish_with_message("Compilation successful!".to_string());
                            compilation_pb.set_prefix("✅");

                            let running_pb = multi_progress.add(ProgressBar::new_spinner());
                            running_pb.set_style(spinner_style.clone());
                            running_pb.set_prefix("🚀");
                            running_pb.set_message("Running tests...".to_string());
                            running_pb.enable_steady_tick(Duration::from_millis(80));

                            compilation_pb = running_pb;
                        }
                        Some("test_result") => {
                            if test_progress.is_none() && json["totalTests"].is_number() {
                                _total_tests = json["totalTests"].as_u64().unwrap_or(0) as usize;

                                if compilation_pb.is_finished() {
                                    compilation_pb.finish();
                                }

                                let progress =
                                    multi_progress.add(ProgressBar::new(_total_tests as u64));
                                progress.set_style(progress_style.clone());
                                progress.set_prefix("🧪 Tests");
                                test_progress = Some(progress);
                            }

                            if let Some(result) = json["result"].as_object() {
                                let test_name = result["name"].as_str().unwrap_or("Unknown test");
                                let status = result["status"].as_str().unwrap_or("UNKNOWN");

                                test_results.push(json["result"].clone());

                                completed_tests += 1;
                                if let Some(ref progress) = test_progress {
                                    let status_symbol =
                                        if status == "PASSED" { "✅" } else { "❌" };
                                    progress.set_position(completed_tests as u64);
                                    progress
                                        .set_message(format!("{} {}", status_symbol, test_name));
                                }
                            }
                        }
                        Some("complete") => {
                            if let Some(progress) = test_progress.take() {
                                progress.finish_and_clear();
                            }
                            compilation_pb.finish_and_clear();

                            let passed = json["passed"].as_bool().unwrap_or(false);
                            let passed_tests = json["passed_tests"].as_u64().unwrap_or(0);
                            let total = json["total_tests"].as_u64().unwrap_or(0);
                            let early_exit = json["early_exit"].as_bool().unwrap_or(false);

                            multi_progress.clear().unwrap();
                            std::thread::sleep(Duration::from_millis(500));

                            let header = if passed {
                                style("✨ ALL TESTS PASSED! ✨").green().bold()
                            } else {
                                style("⚠️ TESTS FAILED ⚠️").red().bold()
                            };

                            println!("{}", header);
                            println!("{}", style("═".repeat(50)).dim());
                            println!("Tests: {}/{} passed", passed_tests, total);
                            println!("{}", style("═".repeat(50)).dim());

                            if early_exit {
                                let reason = json["reason"].as_str().unwrap_or("Unknown reason");
                                println!("\n{}", style("Testing stopped early:").yellow().bold());
                                println!("{}", reason);
                            }
                            println!("{}", style("═".repeat(50)).dim());

                            println!("\n{}", style("Test Results:").bold().underlined());
                            if let Some(results) = json["test_results"].as_array() {
                                for (_, result) in results.iter().enumerate() {
                                    let test_id = result["test_id"].as_u64().unwrap_or(0);
                                    let test_name =
                                        result["name"].as_str().unwrap_or("Unknown test");
                                    let status = result["status"].as_str().unwrap_or("UNKNOWN");

                                    let status_style = if status == "PASSED" {
                                        style(status).green().bold()
                                    } else {
                                        style(status).red().bold()
                                    };

                                    println!("{}. {} - {}", test_id, test_name, status_style);

                                    if status == "FAILED" {
                                        if let Some(debug_info) = result["debug_info"].as_object() {
                                            println!("   {}", style("Debug info:").yellow());
                                            for (key, value) in debug_info {
                                                let formatted_value = if value.is_f64() {
                                                    format!("{:.6}", value.as_f64().unwrap())
                                                } else {
                                                    value.to_string()
                                                };
                                                println!(
                                                    "   - {}: {}",
                                                    style(key).cyan(),
                                                    formatted_value
                                                );
                                            }
                                        }
                                    }
                                }
                            } else {
                                for (i, result) in test_results.iter().enumerate() {
                                    let test_name =
                                        result["name"].as_str().unwrap_or("Unknown test");
                                    let status = result["status"].as_str().unwrap_or("UNKNOWN");

                                    let status_style = if status == "PASSED" {
                                        style(status).green().bold()
                                    } else {
                                        style(status).red().bold()
                                    };

                                    println!("{}. {} - {}", i + 1, test_name, status_style);
                                }
                            }

                            println!("\n{}", style("═".repeat(50)).dim());
                        }
                        _ => {
                            println!("{}", line);
                        }
                    }
                }
            }
        }
    }
}

pub fn pretty_print_benchmark_response(mut response: impl Read) {
    let multi_progress = MultiProgress::new();

    let spinner_style = ProgressStyle::default_spinner()
        .template("{spinner:.green} {prefix:.bold.dim} {wide_msg}")
        .unwrap()
        .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏");

    let progress_style = ProgressStyle::default_bar()
        .template("{prefix:.bold.blue} [{bar:40.blue/cyan}] {pos}/{len} {msg}")
        .unwrap()
        .progress_chars("█▓▒░  ");

    let mut compilation_pb = multi_progress.add(ProgressBar::new_spinner());
    compilation_pb.set_style(spinner_style.clone());
    compilation_pb.set_prefix("🔧");
    compilation_pb.enable_steady_tick(Duration::from_millis(80));

    let mut _total_benchmarks = 0;
    let mut completed_benchmarks = 0;
    let mut benchmark_progress: Option<ProgressBar> = None;
    let mut benchmark_results: Vec<Value> = Vec::new();
    let mut benchmark_names: Vec<String> = Vec::new();
    let mut buffer = [0; 1024];
    let mut data_buffer = String::new();

    while let Ok(size) = response.read(&mut buffer) {
        if size == 0 {
            break;
        }

        let chunk = String::from_utf8_lossy(&buffer[0..size]);
        data_buffer.push_str(&chunk);

        while let Some(pos) = data_buffer.find('\n') {
            let line = data_buffer[..pos].trim().to_string();
            let remaining = data_buffer[pos + 1..].to_string();
            data_buffer = remaining;

            if line.starts_with("data: ") {
                let json_str = &line["data: ".len()..];
                if let Ok(json) = serde_json::from_str::<Value>(json_str) {
                    match json["status"].as_str() {
                        Some("compiling") => {
                            compilation_pb.set_message("Compiling your code...".to_string());
                        }
                        Some("error") => {
                            compilation_pb.finish_with_message("Error detected!".to_string());
                            compilation_pb.set_prefix("❌");

                            compilation_pb.finish_and_clear();
                            multi_progress.clear().unwrap();

                            println!("\n{}", style("⚠️ ERROR OCCURRED ⚠️").red().bold());
                            println!("{}", style("═".repeat(50)).dim());

                            let error_message = json["error"].as_str().unwrap_or("Unknown error");
                            println!("{}: {}", style("Error").red().bold(), error_message);

                            if let Some(details) = json["details"].as_str() {
                                println!("\n{}", style("Details:").yellow().bold());
                                println!("{}", style("─".repeat(50)).dim());

                                let formatted_details = details
                                    .lines()
                                    .map(|line| {
                                        if line.contains("error:") {
                                            format!("{}", style(line).red())
                                        } else if line.contains("warning:") {
                                            format!("{}", style(line).yellow())
                                        } else if line.contains("^") {
                                            format!("{}", style(line).cyan())
                                        } else {
                                            line.to_string()
                                        }
                                    })
                                    .collect::<Vec<String>>()
                                    .join("\n");

                                println!("{}", formatted_details);
                            }

                            println!("{}", style("═".repeat(50)).dim());
                            println!(
                                "{}",
                                style("Please fix the errors and try again.")
                                    .yellow()
                                    .bold()
                            );

                            return;
                        }
                        Some("sanity_check") => {
                            compilation_pb.set_message("Sanity check passed!".to_string());
                            compilation_pb.set_prefix("✅");
                        }
                        Some("running") => {
                            compilation_pb
                                .finish_with_message("Compilation successful!".to_string());
                            compilation_pb.set_prefix("✅");

                            let running_pb = multi_progress.add(ProgressBar::new_spinner());
                            running_pb.set_style(spinner_style.clone());
                            running_pb.set_prefix("🚀");
                            running_pb.set_message("Running benchmarks...".to_string());
                            running_pb.enable_steady_tick(Duration::from_millis(80));

                            compilation_pb = running_pb;
                        }
                        Some("benchmark_result") => {
                            if benchmark_progress.is_none() && json["totalTests"].is_number() {
                                _total_benchmarks =
                                    json["totalTests"].as_u64().unwrap_or(0) as usize;

                                if compilation_pb.is_finished() {
                                    compilation_pb.finish();
                                }

                                let progress =
                                    multi_progress.add(ProgressBar::new(_total_benchmarks as u64));
                                progress.set_style(progress_style.clone());
                                progress.set_prefix("📊 Benchmarks");
                                benchmark_progress = Some(progress);
                            }

                            let name = json["name"].as_str().unwrap_or("Benchmark");
                            benchmark_names.push(name.to_string());
                            if let Some(result) = json["result"].as_object() {
                                let gflops = result["gflops"].as_f64().unwrap_or(0.0);
                                let runtime = result["runtime_ms"].as_f64().unwrap_or(0.0);
                                let status = result["status"].as_str().unwrap_or("UNKNOWN");

                                benchmark_results.push(json["result"].clone());

                                completed_benchmarks += 1;
                                if let Some(ref progress) = benchmark_progress {
                                    let status_symbol =
                                        if status == "PASSED" { "✅" } else { "❌" };
                                    progress.set_position(completed_benchmarks as u64);
                                    progress.set_message(format!(
                                        "{} {:.2} GFLOPS ({:.2} ms)",
                                        status_symbol, gflops, runtime
                                    ));
                                }
                            }
                        }
                        Some("success") => {
                            if let Some(progress) = benchmark_progress.take() {
                                progress.finish_and_clear();
                            }
                            compilation_pb.finish_and_clear();

                            multi_progress.clear().unwrap();
                            std::thread::sleep(Duration::from_millis(500));

                            let avg_gflops = json["gflops"].as_f64().unwrap_or(0.0);
                            let avg_runtime = json["runtime_ms"].as_f64().unwrap_or(0.0);
                            let total = json["total_tests"].as_u64().unwrap_or(0);

                            println!("{}", style("✨ BENCHMARK RESULTS ✨").green().bold());
                            println!("{}", style("═".repeat(65)).dim());
                            println!(
                                "{:<25} {:>15}",
                                style("Metric").bold(),
                                style("Value").bold(),
                            );
                            println!("{}", style("─".repeat(65)).dim());
                            println!("{:<25} {:>15}", "Total Benchmarks:", total);
                            println!("{:<25} {:>15.2}", "Average GFLOPS:", avg_gflops);
                            println!("{:<25} {:>15.2} ms", "Average Runtime:", avg_runtime);
                            println!("{}", style("═".repeat(65)).dim());

                            println!("\n{}", style("Detailed Results:\n").bold().underlined());
                            println!(
                                "{:<18} {:>15} {:>15} {:>15}",
                                style("Test Case").bold(),
                                style("GFLOPS").bold(),
                                style("Runtime (ms)").bold(),
                                style("Std Dev GFLOPS").bold()
                            );
                            println!("{}", style("─".repeat(65)).dim());

                            for (i, result) in benchmark_results.iter().enumerate() {
                                let gflops = result["gflops"].as_f64().unwrap_or(0.0);
                                let runtime = result["runtime_ms"].as_f64().unwrap_or(0.0);
                                let stdev = result["stdev_gflops"].as_f64().unwrap_or(0.0);
                                let status = result["status"].as_str().unwrap_or("UNKNOWN");
                                let name = benchmark_names.get(i).unwrap();

                                let status_indicator = if status == "PASSED" {
                                    style("✓").green().bold()
                                } else {
                                    style("✗").red().bold()
                                };

                                println!(
                                    "{} {:<3} {:<15} {:>12.2} {:>15.4} {:>15.6}",
                                    status_indicator,
                                    i + 1,
                                    name,
                                    gflops,
                                    runtime,
                                    stdev
                                );
                            }

                            if !benchmark_results.is_empty() {
                                println!(
                                    "\n{}",
                                    style("Performance Graph (GFLOPS):").bold().underlined()
                                );
                                println!("{}", style("─".repeat(65)).dim());

                                let max_gflops = benchmark_results
                                    .iter()
                                    .filter_map(|r| r["gflops"].as_f64())
                                    .fold(0.0, |max, val| if val > max { val } else { max });

                                let graph_width = 40;

                                for (i, result) in benchmark_results.iter().enumerate() {
                                    let gflops = result["gflops"].as_f64().unwrap_or(0.0);
                                    let name = benchmark_names.get(i).unwrap();
                                    let bar_length =
                                        ((gflops / max_gflops) * graph_width as f64) as usize;

                                    let max_digits = benchmark_results
                                        .iter()
                                        .filter_map(|r| r["gflops"].as_f64())
                                        .map(|g| format!("{:.2}", g).len())
                                        .max()
                                        .unwrap_or(7);

                                    let label = format!(
                                        "{:<15} ({:.<width$.2}) ",
                                        name,
                                        gflops,
                                        width = max_digits
                                    );

                                    let bar = "█".repeat(bar_length);
                                    let styled_bar = style(bar).green();

                                    println!("{:<15} {}", label, styled_bar);
                                }
                            }

                            println!("\n{}", style("═".repeat(65)).dim());
                            println!(
                                "{}",
                                style("🏁 Benchmark completed successfully! 🏁")
                                    .green()
                                    .bold()
                            );
                        }
                        _ => {} // Catch-all for other status types
                    }
                }
            }
        }
    }
}

pub fn print_parse_error(error: &clap::Error) {
    match error.kind() {
        clap::error::ErrorKind::InvalidValue => {
            let error_message = error.to_string();

            if error_message.contains("PROBLEM_NAME") {
                print_invalid_problem_error(&error_message);
            } else if error_message.contains("GPU_TYPE") {
                print_invalid_gpu_error(&error_message);
            } else if error_message.contains("SOLUTION_FILE") {
                print_invalid_file_error();
            } else if error_message.contains("NOT_SUPPORTED") {
                print_unsupported_file_error();
            } else {
                print_generic_error(&error_message);
            }
        }
        clap::error::ErrorKind::MissingRequiredArgument => {
            print_missing_arg_error(&error.to_string());
        }
        clap::error::ErrorKind::DisplayHelp => {
            print_help(&error.to_string());
        }
        _ => {
            print_generic_error(&error.to_string());
        }
    }
}

fn print_invalid_problem_error(error_message: &str) {
    let problem_value = extract_value_from_error(error_message);
    println!("\n{}", style("⚠️ INVALID PROBLEM NAME ⚠️").red().bold());

    if let Some(value) = problem_value {
        println!(
            "\n{}: '{}'",
            style("Invalid problem name").yellow().bold(),
            value
        );
    }

    println!("\n{}", style("Available problem names:").green().bold());
    println!("{}", style("─".repeat(60)).dim());

    let problems = [
        ("conv-1d", "1D Convolution"),
        ("conv-2d", "2D Convolution"),
        ("gemm-relu", "General Matrix Multiply with ReLU"),
        ("leaky-relu", "Leaky ReLU Activation"),
        ("matrix-multiplication", "Matrix Multiplication"),
        ("matrix-vector", "Matrix-Vector Multiplication"),
        ("relu", "ReLU Activation"),
        ("square-matmul", "Square Matrix Multiplication"),
        ("vector-addition", "Vector Addition"),
    ];

    for (name, desc) in problems {
        println!("  • {} - {}", style(name).cyan().bold(), desc);
    }

    println!(
        "\n{}",
        style("See https://tensara.org/problems for details").yellow()
    );
    println!("{}", style("═".repeat(60)).dim());
}

fn print_invalid_gpu_error(error_message: &str) {
    let gpu_value = extract_value_from_error(error_message);

    println!("\n{}", style("⚠️ INVALID GPU TYPE ⚠️").red().bold());
    println!("{}", style("═".repeat(50)).dim());

    if let Some(value) = gpu_value {
        println!(
            "\n{}: '{}'",
            style("Invalid GPU type").yellow().bold(),
            value
        );
    }

    println!("\n{}", style("Supported GPU types:").green().bold());
    println!("{}", style("─".repeat(50)).dim());

    let gpus = [
        ("T4", "NVIDIA Tesla T4"),
        ("A100", "NVIDIA A100"),
        ("A100_80GB", "NVIDIA A100 80GB"),
        ("H100", "NVIDIA H100"),
        ("L4", "NVIDIA L4"),
        ("L40s", "NVIDIA L40S"),
    ];

    for (name, desc) in gpus {
        println!("  • {} - {}", style(name).cyan().bold(), desc);
    }

    println!("{}", style("═".repeat(50)).dim());
}

fn print_invalid_file_error() {
    println!("\n{}", style("⚠️ INVALID SOLUTION FILE ⚠️").red().bold());
    println!("{}", style("═".repeat(60)).dim());

    println!("\n{}", style("Requirements:").green().bold());
    println!("{}", style("─".repeat(60)).dim());
    println!("  • File must exist");
    println!("  • File must be either a .cu (CUDA) or .py (Python) file");
    println!("  • File must be readable");

    println!("\n{}", style("Example:").yellow().bright().bold());
    println!("  tensara checker -p relu -s ./my_solution.cu");

    println!("{}", style("═".repeat(60)).dim());
}

fn print_missing_arg_error(error_message: &str) {
    println!(
        "\n{}",
        style("⚠️ MISSING REQUIRED ARGUMENT ⚠️").red().bold()
    );
    println!("{}", style("═".repeat(60)).dim());
    println!("{}", style(error_message).red());

    println!("\n{}", style("Usage examples:").green().bold());
    println!("{}", style("─".repeat(60)).dim());
    println!(
        "  • {}",
        style("tensara checker --problem relu --solution ./solution.cu").yellow()
    );
    println!(
        "  • {}",
        style("tensara benchmark -g A100 --problem matrix-vector --solution ./solution.py")
            .yellow()
    );

    println!(
        "\n{}",
        style("Run with --help for more information:").yellow()
    );
    println!("  • {}", style("tensara --help").cyan());
    println!("  • {}", style("tensara checker --help").cyan());
    println!("  • {}", style("tensara benchmark --help").cyan());

    println!("{}", style("═".repeat(60)).dim());
}

fn print_generic_error(error_message: &str) {
    println!("\n{}", style("⚠️ COMMAND ERROR ⚠️").red().bold());
    println!("{}", error_message);

    println!(
        "\n{}",
        style("Try running with --help for more information").yellow()
    );
    println!("{}", style("═".repeat(60)).dim());
}

pub fn print_file_error(file_path: &str, error_message: &str) {
    println!("\n{}", style("⚠️ FILE ERROR ⚠️").red().bold());
    println!("{}", style("═".repeat(60)).dim());
    println!("{}: {}", style("Error").red().bold(), error_message);
    println!("{}: {}", style("File").yellow().bold(), file_path);

    println!("\n{}", style("Make sure:").green().bold());
    println!("  • The file exists");
    println!("  • You have permission to read the file");
    println!("  • The file path is correct");

    println!("{}", style("═".repeat(60)).dim());
}

fn extract_value_from_error(error_message: &str) -> Option<String> {
    let parts: Vec<&str> = error_message.split(':').collect();
    if parts.len() >= 3 {
        return Some(parts[2].trim().to_string());
    }
    None
}

pub fn print_welcome_message() {
    println!("\n{}", style("✨ WELCOME TO TENSARA ✨").blue().bold());
    println!("{}", style("═".repeat(60)).dim());

    println!("\n{}", style("About:").blue().bold());
    println!("A CLI tool for submitting and benchmarking solutions to GPU programming problems.");
    println!(
        "Find available problems at: {}",
        style("https://tensara.org/problems").yellow()
    );

    println!("\n{}", style("Available Commands:").blue().bold());
    println!("{}", style("─".repeat(60)).dim());
    println!(
        "  • {} - {}",
        style("checker").green().bold(),
        "Submit a solution to a problem and check if it is correct"
    );
    println!(
        "  • {} - {}",
        style("benchmark").green().bold(),
        "Benchmark a solution and get performance metrics"
    );

    println!("\n{}", style("Example Usage:").blue().bold());
    println!("{}", style("─".repeat(60)).dim());
    println!(
        "  • {}",
        style("tensara checker -p conv-1d -s solution.cu")
            .yellow()
            .bright()
    );
    println!(
        "  • {}",
        style("tensara benchmark -g A100 -p matrix-multiplication -s solution.py")
            .yellow()
            .bright()
    );

    println!(
        "  • {}",
        style("tensara benchmark -g A100 --problem matrix-vector --solution ./solution.py")
            .yellow()
            .bright()
    );

    println!("\n{}", style("For Help:").blue().bold());
    println!("{}", style("─".repeat(60)).dim());
    println!("  • {}", style("tensara --help").yellow());
    println!("  • {}", style("tensara checker --help").yellow());
    println!("  • {}", style("tensara benchmark --help").yellow());

    println!("{}", style("═".repeat(60)).dim());
}

fn print_help(error_message: &str) {
    println!("{}", style("═".repeat(60)).dim());
    println!("{}", error_message);
    println!("{}", style("═".repeat(60)).dim());
}

fn print_unsupported_file_error() {
    println!("{}", style("═".repeat(60)).dim());
    println!(
        "{}",
        style("We are actively working on enabling Triton support! Please check tensara.org for updates.")
            .green()
            .bold()
    );
    println!("{}", style("═".repeat(60)).dim());
}
