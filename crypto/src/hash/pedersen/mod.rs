use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::short_weierstrass::{
        curves::stark_curve::StarkCurve, point::ShortWeierstrassProjectivePoint,
    },
    field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    },
};

mod constants;
mod parameters;
use self::parameters::PedersenParameters;

pub struct Pedersen {
    params: PedersenParameters,
}

impl Default for Pedersen {
    fn default() -> Self {
        let pedersen_stark_default_params = PedersenParameters::default();
        Self::new_with_params(pedersen_stark_default_params)
    }
}

impl Pedersen {
    pub fn new_with_params(params: PedersenParameters) -> Self {
        Self { params }
    }

    // Taken from Jonathan Lei's starknet-rs
    // https://github.com/xJonathanLEI/starknet-rs/blob/4ab2f36872435ce57b1d8f55856702a6a30f270a/starknet-crypto/src/pedersen_hash.rs
    pub fn hash(
        &self,
        x: &FieldElement<Stark252PrimeField>,
        y: &FieldElement<Stark252PrimeField>,
    ) -> FieldElement<Stark252PrimeField> {
        let x = x.to_bits_le();
        let y = y.to_bits_le();
        let mut acc = self.params.shift_point.clone();

        self.add_points(&mut acc, &x[..248], &self.params.points_p1); // Add a_low * P1
        self.add_points(&mut acc, &x[248..252], &self.params.points_p2); // Add a_high * P2
        self.add_points(&mut acc, &y[..248], &self.params.points_p3); // Add b_low * P3
        self.add_points(&mut acc, &y[248..252], &self.params.points_p4); // Add b_high * P4

        *acc.to_affine().x()
    }

    #[inline]
    fn add_points(
        &self,
        acc: &mut ShortWeierstrassProjectivePoint<StarkCurve>,
        bits: &[bool],
        prep: &[ShortWeierstrassProjectivePoint<StarkCurve>],
    ) {
        bits.chunks(self.params.curve_const_bits)
            .enumerate()
            .for_each(|(i, v)| {
                let offset = bools_to_usize_le(v);
                if offset > 0 {
                    // Table lookup at 'offset-1' in table for chunk 'i'
                    *acc = acc.operate_with(&prep[i * self.params.table_size + offset - 1]);
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

    // Test case ported from:
    // https://github.com/starkware-libs/crypto-cpp/blob/95864fbe11d5287e345432dbe1e80dea3c35fc58/src/starkware/crypto/ffi/crypto_lib_test.go

    #[test]
    fn test_stark_curve() {
        let pedersen = Pedersen::default();

        let x = FieldElement::<Stark252PrimeField>::from_hex_unchecked(
            "03d937c035c878245caf64531a5756109c53068da139362728feb561405371cb",
        );
        let y = FieldElement::<Stark252PrimeField>::from_hex_unchecked(
            "0208a0a10250e382e1e4bbe2880906c2791bf6275695e02fbbc6aeff9cd8b31a",
        );
        let hash = pedersen.hash(&x, &y);
        assert_eq!(
            hash,
            FieldElement::<Stark252PrimeField>::from_hex_unchecked(
                "030e480bed5fe53fa909cc0f8c4d99b8f9f2c016be4c41e13a4848797979c662"
            )
        );
    }
}
