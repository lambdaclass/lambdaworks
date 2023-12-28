#![cfg_attr(not(feature = "std"), no_std)]

pub mod cyclic_group;
pub mod elliptic_curve;
pub mod errors;
pub mod field;
pub mod helpers;
pub mod traits;
pub mod unsigned_integer;

pub mod gpu;

#[cfg(feature = "alloc")]
#[cfg_attr(feature = "alloc", macro_use)]
extern crate alloc;

// These modules don't work in no-std mode
#[cfg(feature = "std")]
pub mod fft;
#[cfg(feature = "alloc")]
pub mod msm;
#[cfg(feature = "alloc")]
pub mod polynomial;
