#[cfg(feature = "cuda")]
fn compile_cuda_shaders() {
    use std::process::Command;

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = std::env::var("OUT_DIR").unwrap();

    let source = format!(
        "{}/src/merkle_tree/cuda/shaders/merkle_tree.cu",
        manifest_dir
    );
    let output = format!("{}/merkle_tree.ptx", out_dir);

    println!("cargo:rerun-if-changed=src/merkle_tree/cuda/shaders/");

    let result = Command::new("nvcc")
        .arg("-ptx")
        .arg(&source)
        .arg("-o")
        .arg(&output)
        .output()
        .expect("Failed to run nvcc - is CUDA toolkit installed?");

    if !result.status.success() {
        eprintln!("{}", String::from_utf8_lossy(&result.stderr));
        panic!("CUDA shader compilation failed");
    }
}

fn main() {
    #[cfg(feature = "cuda")]
    compile_cuda_shaders();
}
