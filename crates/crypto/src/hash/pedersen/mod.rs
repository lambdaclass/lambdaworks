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

// FIXME: currently we make some assumptions that apply to `Stark252PrimeField`, so only mark the
// implementation when that's the field.
impl<P: PedersenParameters<F = Stark252PrimeField>> Pedersen for P {
    // Taken from Jonathan Lei's starknet-rs
    // https://github.com/xJonathanLEI/starknet-rs/blob/4ab2f36872435ce57b1d8f55856702a6a30f270a/starknet-crypto/src/pedersen_hash.rs

    fn hash(x: &FE<Self::F>, y: &FE<Self::F>) -> FE<Self::F> {
        let x = x.to_bits_le();
        let y = y.to_bits_le();
        let mut acc = P::SHIFT_POINT.clone();

        Self::lookup_and_accumulate(&mut acc, &x[..248], &P::POINTS_P1); // Add a_low * P1
        Self::lookup_and_accumulate(&mut acc, &x[248..252], &P::POINTS_P2); // Add a_high * P2
        Self::lookup_and_accumulate(&mut acc, &y[..248], &P::POINTS_P3); // Add b_low * P3
        Self::lookup_and_accumulate(&mut acc, &y[248..252], &P::POINTS_P4); // Add b_high * P4

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
    let mut result: usize = 0;
    for (ind, bit) in bools.iter().enumerate() {
        if *bit {
            result += 1 << ind;
        }
    }
    result
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
