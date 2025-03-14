use tensara::parser;

fn main() {
    let command_matches = parser::get_matches();
    println!("{:?}", command_matches);
}
