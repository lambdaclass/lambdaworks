use core::marker::PhantomData;

use crate::{
    cyclic_group::IsGroup,
    elliptic_curve::short_weierstrass::{
        point::ShortWeierstrassProjectivePoint, traits::IsShortWeierstrass,
    },
    field::traits::IsPrimeField,
    traits::ByteConversion,
    unsigned_integer::traits::IsUnsignedInteger,
};

use rayon::iter::IntoParallelRefIterator;
#[cfg(feature = "parallel")]
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

pub struct Wnaf<EC, ScalarField, const NUM_OF_SCALARS: u32>
where
    EC: IsShortWeierstrass<PointRepresentation = ShortWeierstrassProjectivePoint<EC>>,
    EC::PointRepresentation: Send + Sync,
    ScalarField: IsPrimeField,
{
    table: Vec<Vec<ShortWeierstrassProjectivePoint<EC>>>,
    window_size: usize,
    phantom: PhantomData<ScalarField>,
}

impl<EC, ScalarField, const NUM_OF_SCALARS: u32> Wnaf<EC, ScalarField, NUM_OF_SCALARS>
where
    EC: IsShortWeierstrass<PointRepresentation = ShortWeierstrassProjectivePoint<EC>>,
    EC::PointRepresentation: Send + Sync,
    ScalarField: IsPrimeField + Sync,
{
    pub fn new(base: ShortWeierstrassProjectivePoint<EC>) -> Self {
        let scalar_field_bit_size = ScalarField::field_bit_size();

        let window = Self::get_mul_window_size();
        let in_window = 1 << window;
        let outerc = (scalar_field_bit_size + window - 1) / window;
        let last_in_window = 1 << (scalar_field_bit_size - (outerc - 1) * window);

        let mut g_outer = base;
        let mut g_outers = Vec::with_capacity(outerc);
        for _ in 0..outerc {
            g_outers.push(g_outer.clone()); // performance?
            for _ in 0..window {
                g_outer = g_outer.double();
            }
        }

        let mut table =
            vec![vec![ShortWeierstrassProjectivePoint::<EC>::neutral_element(); in_window]; outerc];

        #[cfg(feature = "parallel")]
        let iter = table.par_iter_mut();
        #[cfg(not(feature = "parallel"))]
        let iter = table.iter_mut();

        iter.enumerate().take(outerc).zip(g_outers).for_each(
            |((outer, multiples_of_g), g_outer)| {
                let curr_in_window = if outer == outerc - 1 {
                    last_in_window
                } else {
                    in_window
                };

                let mut g_inner = ShortWeierstrassProjectivePoint::<EC>::neutral_element();
                for inner in multiples_of_g.iter_mut().take(curr_in_window) {
                    *inner = g_inner.clone(); // performance?
                    g_inner = g_inner.operate_with(&g_outer);
                }
            },
        );

        Self {
            table,
            window_size: window,
            phantom: PhantomData,
        }
    }

    pub fn multi_scalar_mul<T>(&self, v: &[T]) -> Vec<ShortWeierstrassProjectivePoint<EC>>
    where
        T: IsUnsignedInteger + ByteConversion + Sync,
    {
        v.par_iter().map(|e| self.windowed_mul(e.clone())).collect()
    }

    fn windowed_mul<T>(&self, scalar: T) -> ShortWeierstrassProjectivePoint<EC>
    where
        T: IsUnsignedInteger + ByteConversion,
    {
        let mut res = self.table[0][0].clone();

        let modulus_size = ScalarField::field_bit_size();
        let outerc = (modulus_size + self.window_size - 1) / self.window_size;
        let scalar_bits_le: Vec<bool> = scalar
            .to_bytes_le()
            .iter()
            .map(|byte| (0..8).map(|i| (byte >> i) & 1 == 1).collect::<Vec<_>>())
            .flatten()
            .collect();

        for outer in 0..outerc {
            let mut inner = 0usize;
            for i in 0..self.window_size {
                if outer * self.window_size + i < modulus_size
                    && scalar_bits_le[outer * self.window_size + i]
                {
                    inner |= 1 << i;
                }
            }
            res = res.operate_with(&self.table[outer][inner]);
        }

        res.to_affine()
    }

    fn get_mul_window_size() -> usize {
        let scalar_field_bit_size = ScalarField::field_bit_size();
        if scalar_field_bit_size < 32 {
            3
        } else {
            (scalar_field_bit_size as f64).ln().ceil() as usize
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        elliptic_curve::{
            short_weierstrass::curves::bls12_381::{curve::BLS12381Curve, default_types::FrField},
            traits::IsEllipticCurve,
        },
        unsigned_integer::element::U256,
    };
    use rand::*;

    #[test]
    fn anal() {
        let g1 = BLS12381Curve::generator();
        let wnaf = Wnaf::<BLS12381Curve, FrField, 100>::new(g1.clone());

        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(9001);
        let mut scalars = Vec::new();
        for _i in 0..100 {
            scalars.push(U256::from(rng.gen::<u128>()));
        }

        let res1: Vec<ShortWeierstrassProjectivePoint<BLS12381Curve>> = scalars
            .iter()
            .map(|scalar| g1.operate_with_self(scalar.clone()).to_affine())
            .collect();

        let res2 = wnaf.multi_scalar_mul(&scalars);

        for i in 0..100 {
            assert_eq!(res1[i], res2[i]);
        }
    }
}
