#[cfg(feature = "cuda")]
fn compile_cuda_shaders() {
    use std::env;
    use std::process::Command;
    use walkdir::WalkDir;
    const CUDA_SOURCE_DIR: &str = "src/cuda/shaders";

    println!("cargo:rerun-if-changed={CUDA_SHADERS_DIR}");

    let nvcc = option_env!("NVCC_PATH").unwrap_or("nvcc");
    let nvcc_out_dir = env::var("OUT_DIR").unwrap() + "/cuda/";

    let children: Vec<_> = WalkDir::new(CUDA_SHADERS_DIR)
        .into_iter()
        .map(Result::unwrap)
        .filter(|entry| {
            entry
                .path()
                .extension()
                .map(|x| x == "cu")
                .unwrap_or_default()
        })
        .map(|entry| {
            let filename = entry.path().file_stem().unwrap();
            let out_path = nvcc_out_dir.clone() + &filename.to_str().unwrap() + ".ptx";
            println!("cargo:warning=compiling:{out_path}");

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

const METAL_SHADERS_DIR: &str = "src/cuda/shaders";

fn main() {
    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed={METAL_SHADERS_DIR}");

    // println!("cargo:rustc-env=VAR=VALUE");
    #[cfg(feature = "cuda")]
    compile_cuda_shaders();
}
