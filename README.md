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

Run:
``` bash
tensara
```
or 
``` bash
tensara --help
```

For runnning tests or benchmarking:
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

## Uninstall
Remove the binary
```bash
sudo rm /usr/local/bin/tensara
```
