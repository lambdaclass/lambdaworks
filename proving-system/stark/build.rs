use uniffi;

fn main() {
    uniffi::generate_scaffolding("./src/lambdaworks_stark.udl").unwrap();
}
