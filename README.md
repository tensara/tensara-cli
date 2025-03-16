# CLI for Tensara submissions (Beta)

Currently in beta. If you would like to test it, go to modal.com and spin up an endpoint using the cli-engine repository. Put that endpoint in a .env file (see example) and you should be able to run the checker and benchmark

## Install
For Linux/Mac:
```bash
curl -sSL https://tensara.github.io/tensara-cli/install.sh | bash
```

For Windows (untested):
```bash
iwr -useb https://tensara.github.io/tensara-cli/install.ps1 | iex
```

## Usage

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

Supports the same languages and GPUS as the Tensara engine.


You can see this Loom video for a run of the checker and benchmark: [Demo](https://www.loom.com/share/72feb4242b504039b434fefa1b8b8d1e?sid=96e1dbf7-ac91-4ee2-9c93-9cbd664c9e92)
