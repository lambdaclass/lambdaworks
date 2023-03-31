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
}
