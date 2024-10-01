use crate::circle::point::{CirclePoint, HasCircleParams};
use crate::field::traits::IsField;
use crate::field::fields::mersenne31::field::Mersenne31Field;

struct Coset {
    // Coset: shift + <g_n> where n = 2^{log_2_size}.
    // Example: g_16 + <g_8>, n = 8, log_2_size = 3, shift = g_16.
    log_2_size: u128,
    shift: CirclePoint<Mersenne31Field>,
}

impl Coset {
    pub fn new(log_2_size: u128, shift: CirclePoint<Mersenne31Field>) -> Self {
        Coset{ log_2_size, shift }
    }

    /// Returns the coset g_2n + <g_n>
    pub fn new_standard(log_2_size: u128) -> Self {
        // shift is a generator of the subgroup of order 2n = 2^{log_2_size + 1}.
        // We are using that g * k is a generator of the subgroup of order 2^{32 - k}, with k = log_2_size + 1.
        let shift = CirclePoint::generator().mul(31 - log_2_size);
        Coset{ log_2_size, shift }
    }

    /// Given a standard coset g_2n + <g_n>, returns the subcoset with half size g_2n + <g_{n/2}> 
    pub fn half_coset(coset: Self) -> Self {
        Coset { log_2_size: coset.log_2_size + 1, shift: coset.shift }
    }  

    /// Given a coset shift + G returns the coset -shift + G.
    /// Note that (g_2n + <g_{n/2}>) U (-g_2n + <g_{n/2}>) = g_2n + <g_n>.
    pub fn conjugate(coset: Self) -> Self {
        Coset { log_2_size: coset.log_2_size, shift: coset.shift.conjugate() }
    }
}
