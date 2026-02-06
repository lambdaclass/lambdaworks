#[cfg(feature = "cuda")]
fn compile_cuda_shaders() {
    use std::process::Command;
    use walkdir::WalkDir;
    let source_dir = "../math/src/gpu/cuda/shaders";

    // Tell cargo to invalidate the built crate whenever the source changes
    println!("cargo:rerun-if-changed={source_dir}");

    let children: Vec<_> = WalkDir::new(source_dir)
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

/// Compiles Metal shaders (.metal) to a Metal library (.metallib)
/// Uses xcrun to invoke the Metal compiler toolchain.
///
/// This function is only called on macOS where xcrun is guaranteed to be available
/// when Xcode Command Line Tools are installed. Panics are acceptable here because
/// build scripts should fail fast if the toolchain is misconfigured.
#[cfg(feature = "metal")]
fn compile_metal_shaders() {
    use std::env;
    use std::path::Path;
    use std::process::Command;

    let source_dir = "../math/src/gpu/metal";
    let source_file = format!("{}/all.metal", source_dir);
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let output_file = format!("{}/lib.metallib", out_dir);

    // Tell cargo to invalidate the built crate whenever the source changes
    println!("cargo:rerun-if-changed={source_dir}");

    if !cfg!(target_os = "macos") {
        std::fs::write(&output_file, []).expect("failed to write placeholder metallib");
        println!("cargo:warning=Metal shaders are only compiled on macOS; using empty metallib");
        return;
    }

    // Check if source file exists - skip compilation if shaders haven't been created yet
    if !Path::new(&source_file).exists() {
        std::fs::write(&output_file, []).expect("failed to write placeholder metallib");
        println!("cargo:warning=Metal source file not found: {}", source_file);
        println!("cargo:warning=Skipping Metal shader compilation - create shaders first");
        return;
    }

    println!(
        "cargo:warning=Compiling Metal shaders: '{}' -> '{}'",
        source_file, output_file
    );

    // Compile .metal to .air (intermediate representation)
    let air_file = format!("{}/all.air", out_dir);
    let metal_compile = Command::new("xcrun")
        .args([
            "-sdk",
            "macosx",
            "metal",
            "-c",
            &source_file,
            "-o",
            &air_file,
        ])
        .output();

    let metal_compile = match metal_compile {
        Ok(output) => output,
        Err(e) => {
            std::fs::write(&output_file, []).expect("failed to write placeholder metallib");
            println!("cargo:warning=Metal compiler not available ({}), using empty metallib", e);
            println!("cargo:warning=Install Xcode (not just Command Line Tools) to compile Metal shaders");
            return;
        }
    };

    if !metal_compile.status.success() {
        let stderr = String::from_utf8_lossy(&metal_compile.stderr);
        if stderr.contains("unable to find utility") {
            std::fs::write(&output_file, []).expect("failed to write placeholder metallib");
            println!("cargo:warning=Metal compiler not found via xcrun, using empty metallib");
            println!("cargo:warning=Install Xcode (not just Command Line Tools) to compile Metal shaders");
            return;
        }
        eprintln!("Metal compilation failed:\n{}", stderr);
        panic!("Metal shader compilation failed - check shader syntax");
    }

    // Link .air to .metallib
    // Panics are acceptable in build scripts when the toolchain is missing
    let metallib_link = Command::new("xcrun")
        .args(["-sdk", "macosx", "metallib", &air_file, "-o", &output_file])
        .output()
        .expect("xcrun metallib linker not found - install Xcode Command Line Tools");

    if !metallib_link.status.success() {
        eprintln!(
            "Metal linking failed:\n{}",
            String::from_utf8_lossy(&metallib_link.stderr)
        );
        panic!("Metal library linking failed");
    }

    // Clean up intermediate .air file - ignore errors as this is just cleanup
    let _ = std::fs::remove_file(&air_file);

    println!("cargo:warning=Metal shaders compiled successfully");
}

fn main() {
    #[cfg(feature = "cuda")]
    compile_cuda_shaders();

    #[cfg(feature = "metal")]
    compile_metal_shaders();
}
