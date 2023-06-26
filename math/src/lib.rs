#![cfg_attr(not(feature = "std"), no_std)]

pub mod cyclic_group;
pub mod errors;
pub mod field;
pub mod helpers;
pub mod traits;
pub mod unsigned_integer;

pub mod gpu;

// These modules don't work in no-std mode
#[cfg(feature = "std")]
pub mod elliptic_curve;
#[cfg(feature = "std")]
pub mod fft;
#[cfg(feature = "std")]
pub mod msm;
#[cfg(feature = "std")]
pub mod polynomial;
