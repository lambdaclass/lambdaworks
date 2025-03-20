#[cfg(all(feature = "metal", feature = "cuda"))]
compile_error!(
    "Can't enable both \"metal\" and \"cuda\" features at the same time.
If you were using the `--all-features` flag please read this crate's Cargo.toml"
);

#[cfg(feature = "metal")]
pub mod metal;

#[cfg(feature = "cuda")]
pub mod cuda;
