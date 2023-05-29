use std::env;
use std::process::Command;
use walkdir::WalkDir;

const CUDA_SHADERS_DIR: &str = "src/cuda/shaders";
const METAL_SHADERS_DIR: &str = "src/cuda/shaders";

fn main() {
    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed={CUDA_SHADERS_DIR}");
    println!("cargo:rerun-if-changed={METAL_SHADERS_DIR}");

    // println!("cargo:rustc-env=VAR=VALUE");

    let nvcc = option_env!("NVCC_PATH").unwrap_or("nvcc");
    let nvcc_out_dir = env::var("OUT_DIR").unwrap() + "/nvcc/";

    let children: Vec<_> = WalkDir::new(CUDA_SHADERS_DIR)
        .into_iter()
        .map(Result::unwrap)
        .filter(|entry| entry.path().ends_with(".cu"))
        .map(|entry| {
            let filename = entry.file_name().to_str().unwrap();
            let out_path = nvcc_out_dir.clone() + &filename[..filename.len() - 3] + ".ptx";

            Command::new(nvcc)
                .arg("-ptx")
                .arg(entry.path())
                .arg("-o")
                .arg(out_path)
                .spawn()
                .unwrap()
        })
        .collect();

    children.into_iter().for_each(|mut child| {
        let status = child.wait().unwrap();
        if !status.success() {
            panic!("Failed to compile cuda shader");
        }
    });
}
