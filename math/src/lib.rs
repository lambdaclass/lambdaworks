#![cfg_attr(not(feature = "std"), no_std)]
#[cfg(feature = "alloc")]
extern crate alloc;

pub mod cyclic_group;
pub mod errors;
pub mod field;
pub mod helpers;
pub mod traits;
pub mod unsigned_integer;

pub mod gpu;

#[cfg(feature = "alloc")]
pub mod polynomial;
// These modules don't work in no-std mode
#[cfg(feature = "std")]
pub mod elliptic_curve;
#[cfg(feature = "std")]
pub mod fft;
#[cfg(feature = "std")]
pub mod msm;
