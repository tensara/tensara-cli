# CLI for Tensara submissions (Beta)

Currently in beta. Allow users to practice Tensara problems from th e comfort of their own IDE.
For now, CLI submissions do not show up on the leaderboard. Actively working on this, expect updates in 2 weeks!

## Install
For Linux/Mac:
```bash
curl -sSL https://get.tensara.org/install.sh | bash
```

For Windows (untested):
```bash
iwr -useb https://get.tensara.org/install.sh | iex
```

## Usage

Run:
```bash
tensara
```

or
``` bash
tensara --help
```

For running tests or benchmarking:
```bash
tensara (checker | benchmark) -g <gpu_type> --problem <problem_name> --solution <solution_file>
```

## Example

```bash
tensara checker -g T4 --problem vector-addition --solution tests/sol.cu
```

Short forms for args are also supported:

```bash
tensara checker -g T4 -p vector-addition -s tests/sol.cu
```

Supports the same languages and GPUs as the Tensara engine.

## View Problems

To view available problems, use:
```bash
tensara problems
```

You can show different fields by using the `--fields` flag and sort by `--sort-by`:
```bash
tensara problems --f tags --s difficulty
```

To see all available flags:
```bash
tensara problems --help
```

## Authenticate

Before submitting solutions, authenticate using your Tensara account:
```bash
tensara auth -t <token>
```
For more details, visit [tensara.org/cli](https://tensara.org/cli).

## Init 

To initialize a new project, use:
```bash
tensara init <directory> -p <problem_name> -l <language>
```
This will create a template solution file and a problem file in the specified directory.


## Submit a Solution

To submit your solution on Tensara, use the `submit` command:
```bash
tensara submit -g <gpu_type> -p <problem_name> -s <solution_file>
```

Example:
```bash
tensara submit -g T4 -p vector-addition -s solution.cu
```

Your results will be shown in the CLI. Submissions will be reflected on the Tensara dashboard.



## Uninstall

Remove the binary:
```bash
sudo rm /usr/local/bin/tensara
```

---

This documentation will evolve as we expand support. For the latest updates, visit [tensara.org/cli](https://tensara.org/cli).
