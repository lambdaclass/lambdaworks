#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod cyclic_group;
pub mod elliptic_curve;
pub mod errors;
pub mod field;
pub mod helpers;
pub mod traits;
pub mod unsigned_integer;

pub mod gpu;

// These modules don't work in no-std mode
pub mod fft;
pub mod msm;
#[cfg(feature = "alloc")]
pub mod polynomial;
