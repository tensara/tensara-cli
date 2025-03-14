use tensara::parser;

fn main() {
    let command_matches = parser::get_matches();
    let checker_matches = parser::get_checker_matches(&command_matches);
    let problem_name = parser::get_problem_name(&checker_matches);
    let solution_file = parser::get_solution_file(&checker_matches);
    let gpu_type = parser::get_gpu_type(&checker_matches);

    println!("Problem name: {}", problem_name);
    println!("Solution file: {}", solution_file);
    println!("GPU type: {}", gpu_type);
}
