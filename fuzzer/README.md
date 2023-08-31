## Setup
Run the following commands to get ready the setup.

* `cargo install honggfuzz `
* `apt install build-essential`
* `apt-get install binutils-dev`
* `sudo apt-get install libunwind-dev`
* `sudo apt-get install lldb`

## Run the fuzzer 

Run the following command to run the specific fuzzer
`cargo hfuzz run <name of the fuzz target> `
