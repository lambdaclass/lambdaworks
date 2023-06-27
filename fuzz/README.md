# Fuzzing

### Running the fuzzers
`cargo +nightly fuzz run <target_name>`
The targets can be found in the `fuzz_targets` directory. Normally the name of the file without the extension should work, if it doesn't, look up the name for that binary target in `Cargo.toml`.

### Debugging
If a crash is found, an `artifacts/<target_name>` folder will be added, inside it you'll find the different reports. To get an lldb dump, run
`cargo +nightly fuzz run <target_name> artifacts/crash-xxx`
