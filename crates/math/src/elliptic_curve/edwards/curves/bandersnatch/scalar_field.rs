//! Scalar field of Bandersnatch curve.
//!
//! This is the order of the prime-order subgroup, used for scalar multiplication.
//! The full curve has order h * r where h = 4 (cofactor) and r is this field's order.
//!
//! Note: This is different from the base field (BLS12-381's Fr).

use crate::unsigned_integer::element::U256;

/// The order of the prime-order subgroup of Bandersnatch.
/// r = 13108968793781547619861935127046491459309155893440570251786403306729687672801
pub const BANDERSNATCH_SUBGROUP_ORDER: U256 =
    U256::from_hex_unchecked("1cfb69d4ca675f520cce760202687600ff8f87007419047174fd06b52876e7e1");
