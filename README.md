# nvsmi

A (user-)friendly wrapper to `nvidia-smi`.

## Usage

### CLI

```
nvsmi --help
nvsmi ls --help
nvsmi ps --help
```

### As a library

```
import nvsmi

nvsmi.get_gpus()
nvsmi.get_available_gpus()
nvsmi.get_gpu_processes()
```

## Prerequisites

- An nvidia GPU
- `nvidia-smi`
- Python 2.7 or 3.6+

## Installation

### pipx

The recommended installation method is [pipx](https://github.com/pipxproject/pipx).
More specifically, you can install `nvsmi` for your user with:

``` shell
pipx install nvsmi
```

The above command will create a virtual environment in `~/.local/pipx/venvs/nvsmi` and
add the `nvsmi` executable in `~/.local/bin`.

### pip

Alternatively you can use good old `pip` but this is more fragile than `pipx`:

```
pip install --user nvsmi
```
