#[cfg(feature = "cuda")]
fn compile_cuda_shaders() {
    use std::env;
    use std::process::Command;
    use walkdir::WalkDir;
    const CUDA_SOURCE_DIR: &str = "src/cuda/shaders";

    // Tell cargo to invalidate the built crate whenever the source changes
    println!("cargo:rerun-if-changed={CUDA_SOURCE_DIR}");

    let nvcc = option_env!("CUDA_BIN").unwrap_or("nvcc");
    let nvcc_out_dir = env::var("OUT_DIR").unwrap() + "/cuda/";

    let children: Vec<_> = WalkDir::new(CUDA_SOURCE_DIR)
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

#[cfg(feature = "metal")]
fn compile_metal_shaders() {
    use std::process::Command;
    const METAL_SHADERS_DIR: &str = "src/metal/shaders";

    // Tell cargo to invalidate the built crate whenever the source changes
    println!("cargo:rerun-if-changed={METAL_SHADERS_DIR}");

    let input = METAL_SHADERS_DIR.to_owned() + "/all.metal";
    let output = METAL_SHADERS_DIR.to_owned() + "/lib.metallib";

    let cmd = Command::new("xcrun")
        .args(&["-sdk", "macosx", "metal"])
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .spawn()
        .expect("Failed to spawn process");

    let res = cmd.wait_with_output().expect("Command waiting failed");

    if !res.status.success() {
        println!();
        println!("{}", String::from_utf8(res.stdout).unwrap());
        println!();
        eprintln!("{}", String::from_utf8(res.stderr).unwrap());
        println!();
        panic!("Compilation failed for source '{}'", input);
    }
}

fn main() {
    #[cfg(feature = "cuda")]
    compile_cuda_shaders();

    #[cfg(feature = "metal")]
    compile_metal_shaders();
}
