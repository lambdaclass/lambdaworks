// By removing refs as clippy wants
// Implementations with all the combination of reference and not references become recursive
#[allow(clippy::op_ref)]
pub mod element;
pub mod montgomery;
pub mod traits;

// ARM64-specific assembly implementations
#[cfg(target_arch = "aarch64")]
pub mod asm_aarch64;
