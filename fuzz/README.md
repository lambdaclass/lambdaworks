# Fuzzing
There are three types of fuzzers distributed on different workspaces depending on the features (metal/cuda) they need. So you should make sure you cded into the right folder before running any of the commands.
This directory contains three types of fuzzers distributed onto three different workspaces. `no_gpu_fuzz` and `metal_fuzz` both use `cargo-fuzz` and must be run with nightly, also the latter only runs on mac. `cuda_fuzz` runs on machines with nvida GPUs and uses `honggfuzz` which runs on most linux distros.

## Setup
Run the following commands to get ready. 

### cargo-fuzz
* `cargo install cargo-fuzz`
* `rustup toolchain install nightly`

### honggfuzz
* `cargo install honggfuzz`
* `apt install build-essential`
* `apt-get install binutils-dev`
* `sudo apt-get install libunwind-dev`
* `sudo apt-get install lldb`

## Running the fuzzers
* no_gpu & metal: `cargo +nightly fuzz run --fuzz-dir . <target_name>`
* cuda: `cargo hfuzz run <target_name> `
The targets can be found in the `fuzz_targets` directory. Normally the name of the file without the extension should work, if it doesn't, look up the name for that binary target in `Cargo.toml`.

## Debugging
If a crash is found, an `artifacts/<target_name>` or `hfuzz_workspace/cuda_fuzz` folder will be added, inside it you'll find the different reports. To get an lldb dump, run
* no_gpu & metal: `cargo +nightly fuzz run --fuzz-dir . <target_name> artifacts/<crash-xxx>`
* cuda: `cargo hfuzz run-debug <target_name> hfuzz_workspace/cuda_fuzz/<name.fuzz>`
