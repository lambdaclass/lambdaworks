use alloc::vec::Vec;
use lambdaworks_math::field::{
    element::FieldElement,
    traits::{IsFFTField, IsSubFieldOf},
};
use lambdaworks_math::polynomial::Polynomial;

use crate::linear_code_pcs::traits::LinearCodeEncoding;

/// Reed-Solomon encoding backend for the Ligero PCS.
///
/// Treats the message as polynomial coefficients, then evaluates the polynomial
/// at `codeword_len` roots of unity via FFT. This gives an RS codeword of
/// rate `1 / rho_inv`.
///
/// Requires an FFT-friendly field.
#[derive(Clone, Debug)]
pub struct ReedSolomonEncoding<F: IsFFTField> {
    /// Length of the input message (number of polynomial coefficients).
    msg_len: usize,
    /// Inverse of the code rate: `codeword_len = msg_len * rho_inv`.
    rho_inv: usize,
    _marker: core::marker::PhantomData<F>,
}

impl<F: IsFFTField> ReedSolomonEncoding<F> {
    /// Create a new RS encoding for messages of length `msg_len`.
    ///
    /// `rho_inv` is the blowup factor (inverse rate). E.g., `rho_inv = 2` means
    /// rate 1/2 code with distance `1 - 1/2 = 1/2`.
    ///
    /// `msg_len * rho_inv` must be a power of two.
    pub fn new(msg_len: usize, rho_inv: usize) -> Self {
        assert!(rho_inv >= 2, "rho_inv must be at least 2");
        let cw_len = msg_len * rho_inv;
        assert!(
            cw_len.is_power_of_two(),
            "msg_len * rho_inv must be a power of two, got {}",
            cw_len
        );
        Self {
            msg_len,
            rho_inv,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<F> LinearCodeEncoding<F> for ReedSolomonEncoding<F>
where
    F: IsFFTField + IsSubFieldOf<F>,
{
    fn encode(&self, msg: &[FieldElement<F>]) -> Vec<FieldElement<F>> {
        assert_eq!(msg.len(), self.msg_len, "message length mismatch");
        let cw_len = self.msg_len * self.rho_inv;

        // Treat message as coefficients of a polynomial of degree < msg_len.
        // Pad to cw_len and evaluate via FFT at cw_len roots of unity.
        let poly = Polynomial::new(msg);
        Polynomial::evaluate_fft::<F>(&poly, 1, Some(cw_len))
            .expect("FFT evaluation should succeed for valid power-of-two domain")
    }

    fn codeword_len(&self) -> usize {
        self.msg_len * self.rho_inv
    }

    fn message_len(&self) -> usize {
        self.msg_len
    }

    fn distance(&self) -> (usize, usize) {
        // RS distance = 1 - rate = 1 - 1/rho_inv = (rho_inv - 1) / rho_inv
        (self.rho_inv - 1, self.rho_inv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;

    type F = Stark252PrimeField;
    type FE = FieldElement<F>;

    #[test]
    fn encode_produces_correct_length() {
        let enc = ReedSolomonEncoding::<F>::new(4, 2);
        let msg: Vec<FE> = (1..=4).map(|x| FE::from(x as u64)).collect();
        let cw = enc.encode(&msg);
        assert_eq!(cw.len(), 8);
    }

    #[test]
    fn encode_is_systematic_at_roots() {
        // For an RS code, the codeword is the evaluation of the polynomial at
        // roots of unity. Evaluating at w^0 = 1 should give the sum of coefficients.
        let enc = ReedSolomonEncoding::<F>::new(4, 2);
        let msg = vec![
            FE::from(1u64),
            FE::from(2u64),
            FE::from(3u64),
            FE::from(4u64),
        ];
        let cw = enc.encode(&msg);
        // cw[0] = poly(w^0) = poly(1) = 1 + 2 + 3 + 4 = 10
        assert_eq!(cw[0], FE::from(10u64));
    }

    #[test]
    fn distance_rate_half() {
        let enc = ReedSolomonEncoding::<F>::new(4, 2);
        assert_eq!(enc.distance(), (1, 2));
    }

    #[test]
    fn distance_rate_quarter() {
        let enc = ReedSolomonEncoding::<F>::new(4, 4);
        assert_eq!(enc.distance(), (3, 4));
    }
}
