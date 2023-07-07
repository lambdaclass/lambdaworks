# Fuzzing
There are three types of fuzzers distributed on different workspaces depending on the features (metal/cuda) they need. So you should make sure you cded into the right folder before running any of the commands.

### Running the fuzzers
`cargo +nightly fuzz run --fuzz-dir . <target_name>`
The targets can be found in the `fuzz_targets` directory. Normally the name of the file without the extension should work, if it doesn't, look up the name for that binary target in `Cargo.toml`.

### Debugging
If a crash is found, an `artifacts/<target_name>` folder will be added, inside it you'll find the different reports. To get an lldb dump, run
`cargo +nightly fuzz run --fuzz-dir . <target_name> artifacts/crash-xxx`
