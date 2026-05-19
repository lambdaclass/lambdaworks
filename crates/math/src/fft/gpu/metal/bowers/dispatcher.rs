//! Rust dispatcher for the Bowers-G fused LDE Metal kernels (filled in by Task 7).

use lambdaworks_gpu::metal::abstractions::{errors::MetalError, state::MetalState};
use metal::Buffer;

pub fn metal_bowers_lde(
    _cols: &Buffer,
    _twiddles_bitrev: &Buffer,
    _coset_powers: &Buffer,
    _out: &mut Buffer,
    _log_n: u32,
    _num_cols: u32,
    _state: &MetalState,
) -> Result<(), MetalError> {
    unimplemented!("filled in by Task 7")
}

pub fn metal_bowers_lde_fp3(
    _cols: &Buffer,
    _twiddles_bitrev: &Buffer,
    _coset_powers: &Buffer,
    _out: &mut Buffer,
    _log_n: u32,
    _num_cols: u32,
    _state: &MetalState,
) -> Result<(), MetalError> {
    unimplemented!("filled in by Task 12")
}
