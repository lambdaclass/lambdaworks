//! Bandersnatch twisted Edwards curve over BLS12-381's scalar field.
//!
//! See [ePrint 2021/1152](https://eprint.iacr.org/2021/1152).

mod curve;

pub use curve::{
    BandersnatchBaseField, BandersnatchCurve, BANDERSNATCH_COFACTOR, BANDERSNATCH_SUBGROUP_ORDER,
};
