use lambdaworks_math::{
    elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint as Point,
    field::{
        element::FieldElement as FE,
        fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    },
};

mod constants;
mod parameters;
use parameters::PedersenParameters;
pub use parameters::PedersenStarkCurve;

mod private {
    use super::*;

    pub trait Sealed {}

    impl<P: PedersenParameters> Sealed for P {}
}

pub trait Pedersen: PedersenParameters + self::private::Sealed {
    /// Implements Starkware version of Pedersen hash of x and y.
    /// Divides each of x and y into 4-bit chunks, and uses lookup tables to accumulate pre-calculated
    /// points corresponding to a given chunk.
    /// Accumulation starts from a "shift_point" whose points are derived from digits of pi.
    /// Pre-calculated points are multiples by powers of 2 of the "shift_point".
    ///
    /// Find specification at https://docs.starkware.co/starkex/crypto/pedersen-hash-function.html
    fn hash(x: &FE<Self::F>, y: &FE<Self::F>) -> FE<Self::F>;

    /// Performs lookup to find the constant point corresponding to 4-bit chunks of given input.
    /// Keeps adding up those points to the given accumulation point.
    fn lookup_and_accumulate(acc: &mut Point<Self::EC>, bits: &[bool], prep: &[Point<Self::EC>]);
}

/// Constrained to `Stark252PrimeField` because:
/// - `to_bits_le()` is a concrete method on `FieldElement<Stark252PrimeField>`, not a trait method
/// - The Starkware Pedersen hash spec splits 252-bit field elements into a 248-bit low part
///   and a 4-bit high part, matching the lookup table sizes (P1/P3: 930 entries for 62 chunks,
///   P2/P4: 15 entries for 1 chunk, with CURVE_CONST_BITS = 4)
impl<P: PedersenParameters<F = Stark252PrimeField>> Pedersen for P {
    // Taken from Jonathan Lei's starknet-rs
    // https://github.com/xJonathanLEI/starknet-rs/blob/4ab2f36872435ce57b1d8f55856702a6a30f270a/starknet-crypto/src/pedersen_hash.rs

    fn hash(x: &FE<Self::F>, y: &FE<Self::F>) -> FE<Self::F> {
        // Stark252PrimeField elements are 252 bits; the low 248 bits and high 4 bits
        // are processed separately with different precomputed point tables.
        let field_element_bits: usize = 252;
        let low_part_bits: usize = field_element_bits - P::CURVE_CONST_BITS;

        let x = x.to_bits_le();
        let y = y.to_bits_le();
        let mut acc = P::SHIFT_POINT.clone();

        Self::lookup_and_accumulate(&mut acc, &x[..low_part_bits], &P::POINTS_P1);
        Self::lookup_and_accumulate(
            &mut acc,
            &x[low_part_bits..field_element_bits],
            &P::POINTS_P2,
        );
        Self::lookup_and_accumulate(&mut acc, &y[..low_part_bits], &P::POINTS_P3);
        Self::lookup_and_accumulate(
            &mut acc,
            &y[low_part_bits..field_element_bits],
            &P::POINTS_P4,
        );

        *acc.to_affine().x()
    }

    fn lookup_and_accumulate(acc: &mut Point<Self::EC>, bits: &[bool], prep: &[Point<Self::EC>]) {
        bits.chunks(P::CURVE_CONST_BITS)
            .enumerate()
            .for_each(|(i, v)| {
                let offset = bools_to_usize_le(v);
                if offset > 0 {
                    // Table lookup at 'offset-1' in table for chunk 'i'
                    *acc = acc.operate_with_affine(&prep[i * P::TABLE_SIZE + offset - 1]);
                }
            })
    }
}

#[inline]
fn bools_to_usize_le(bools: &[bool]) -> usize {
    bools
        .iter()
        .enumerate()
        .filter(|(_, &bit)| bit)
        .fold(0, |acc, (i, _)| acc | (1 << i))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::pedersen::parameters::PedersenStarkCurve;

    // Test case ported from:
    // https://github.com/starkware-libs/crypto-cpp/blob/95864fbe11d5287e345432dbe1e80dea3c35fc58/src/starkware/crypto/ffi/crypto_lib_test.go

    #[test]
    fn test_stark_curve() {
        let x = FE::from_hex_unchecked(
            "03d937c035c878245caf64531a5756109c53068da139362728feb561405371cb",
        );
        let y = FE::from_hex_unchecked(
            "0208a0a10250e382e1e4bbe2880906c2791bf6275695e02fbbc6aeff9cd8b31a",
        );
        let hash = PedersenStarkCurve::hash(&x, &y);
        assert_eq!(
            hash,
            FE::from_hex_unchecked(
                "030e480bed5fe53fa909cc0f8c4d99b8f9f2c016be4c41e13a4848797979c662"
            )
        );
    }
}
