#[cfg(feature = "cuda")]
fn compile_cuda_shaders() {
    use std::process::Command;
    use walkdir::WalkDir;
    const CUDA_SOURCE_DIR: &str = "src/cuda/shaders";

    // Tell cargo to invalidate the built crate whenever the source changes
    println!("cargo:rerun-if-changed={CUDA_SOURCE_DIR}");

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
            let mut out_path = entry.path().to_owned();
            out_path.set_extension("ptx");

            println!(
                "cargo:warning=compiling:'{}'->'{}'",
                entry.path().display(),
                out_path.display(),
            );

            Command::new("nvcc")
                .arg("-ptx")
                .arg(entry.path())
                .arg("-o")
                .arg(out_path)
                .spawn()
                .unwrap()
        })
        .collect();

    children.into_iter().for_each(|child| {
        let res = child.wait_with_output().unwrap();
        if !res.status.success() {
            println!();
            println!("{}", String::from_utf8(res.stdout).unwrap());
            println!();
            eprintln!("{}", String::from_utf8(res.stderr).unwrap());
            println!();
            panic!("Compilation failed");
        }
    });
}

#[cfg(feature = "metal")]
fn compile_metal_shaders() {
    use std::process::Command;
    const METAL_SOURCE_DIR: &str = "src/metal/shaders";

    // Tell cargo to invalidate the built crate whenever the source changes
    println!("cargo:rerun-if-changed={METAL_SOURCE_DIR}");

    let input = METAL_SOURCE_DIR.to_owned() + "/all.metal";
    let output = METAL_SOURCE_DIR.to_owned() + "/lib.metallib";

    println!("cargo:warning=compiling:'{input}'->'{output}'");

    let cmd = Command::new("xcrun")
        .args(["-sdk", "macosx", "metal"])
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
