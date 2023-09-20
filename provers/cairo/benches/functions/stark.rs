use cairo_platinum_prover::{cairo_mem::CairoMemory, register_states::RegisterStates};

pub fn generate_cairo_trace(filename: &str) -> (RegisterStates, CairoMemory) {
    let base_dir = env!("CARGO_MANIFEST_DIR").to_string() + "/src/cairo_vm/test_data/";

    let trace_path = format!("{base_dir}/{filename}.trace");
    let memory_path = format!("{base_dir}/{filename}.memory");

    let register_states =
        RegisterStates::from_file(&trace_path).expect("Cairo trace binary file not found");
    let memory = CairoMemory::from_file(&memory_path).expect("Cairo memory binary file not found");

    (register_states, memory)
}
