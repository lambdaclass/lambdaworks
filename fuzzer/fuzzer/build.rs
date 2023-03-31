fn build_go_backend() {
    std::process::Command::new("make")
        .arg("-C")
        .arg(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .to_str()
                .unwrap(),
        )
        .arg("build-go")
        .spawn()
        .unwrap();
}

fn main() {
    build_go_backend();

    let path = "gnark_backend_ffi";
    let lib = "gnark_backend";

    println!("cargo:rustc-link-search=native={path}");
    println!("cargo:rustc-link-lib=static={lib}");
}
