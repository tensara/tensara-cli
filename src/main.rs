use tensara::commands;

fn main() {
    let commands = commands::new();
    let command_matches = commands
        .get("checker")
        .expect("Test command is not available");

    println!("{:?}", command_matches);
}
