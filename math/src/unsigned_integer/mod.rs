// By removing refs as clippy wants
// Implementations with all the combination of reference and not references become recursive
#[allow(clippy::op_ref)]
pub mod element;
pub mod montgomery;
pub mod traits;
