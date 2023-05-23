#![allow(unused)]
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::short_weierstrass::{
        curves::bls12_381::{curve::BLS12381Curve, field_extension::BLS12381PrimeField},
        point::ShortWeierstrassProjectivePoint,
    },
    field::{element::FieldElement, traits::IsField},
    unsigned_integer::{element::UnsignedInteger, traits::U32Limbs},
};

use crate::metal::abstractions::{errors::MetalError, state::*};

use metal::{ComputeCommandEncoderRef, MTLSize};

type Point = ShortWeierstrassProjectivePoint<BLS12381Curve>;
pub type F = BLS12381PrimeField;
pub type FE = FieldElement<F>;

/// Computes the multiscalar multiplication (MSM), using Pippenger's algorithm parallelized in
/// Metal.
pub fn pippenger<const NUM_LIMBS: usize>(
    cs: &[UnsignedInteger<NUM_LIMBS>],
    hidings: &[Point],
    window_size: usize,
    state: &MetalState,
) -> Result<Point, MetalError> {
    let point_len = hidings.len(); // == cs.len();
    if point_len == 0 {
        return Ok(Point::neutral_element());
    }

    let n_bits = 64 * NUM_LIMBS;
    let num_windows = (n_bits - 1) / window_size + 1;
    let buckets_len = (1 << window_size) - 1;

    let mut buckets_matrix = vec![Point::neutral_element(); buckets_len * point_len];
    // TODO: make a helper func for converting collections into limbs
    let mut buckets_matrix_limbs: Vec<u32> = buckets_matrix
        .iter()
        .flat_map(|b| b.to_u32_limbs())
        .collect();

    let k_limbs = cs
        .iter()
        .flat_map(|uint| uint.to_u32_limbs())
        .collect::<Vec<_>>();
    let p_limbs = hidings
        .iter()
        .flat_map(|uint| uint.to_u32_limbs())
        .collect::<Vec<_>>();

    let k_buffer = state.alloc_buffer_data(&k_limbs);
    let p_buffer = state.alloc_buffer_data(&p_limbs);
    let wsize_buffer = state.alloc_buffer_data(&[window_size as u32]);

    let org_buckets_pipe = state.setup_pipeline("calculate_buckets").unwrap();
    (0..num_windows)
        .rev()
        .map(|window_idx| {
            let buckets_matrix_buffer = state.alloc_buffer_data(&buckets_matrix_limbs);

            objc::rc::autoreleasepool(|| {
                let (command_buffer, command_encoder) = state.setup_command(
                    &org_buckets_pipe,
                    Some(&[
                        (1, &wsize_buffer),
                        (2, &k_buffer),
                        (3, &p_buffer),
                        (4, &buckets_matrix_buffer),
                    ]),
                );

                MetalState::set_bytes(0, &[window_idx], command_encoder);

                command_encoder.dispatch_thread_groups(
                    MTLSize::new(1, 1, 1),
                    MTLSize::new(point_len as u64, 1, 1),
                );
                command_encoder.end_encoding();
                command_buffer.commit();
                command_buffer.wait_until_completed();
            });

            let mut buckets_matrix: Vec<Point> =
                MetalState::retrieve_contents(&buckets_matrix_buffer)
                    .chunks(12 * 3)
                    .map(Point::from_u32_limbs)
                    .collect();

            let mut buckets = Vec::with_capacity(buckets_len);

            // TODO: use iterators
            for i in 0..buckets_len {
                let mut partial_sum = buckets_matrix[i].clone();

                for j in 1..point_len {
                    partial_sum = partial_sum.operate_with(&buckets_matrix[i + j * buckets_len]);
                }
                buckets.push(partial_sum);
            }

            buckets
                .iter_mut()
                .rev()
                .scan(Point::neutral_element(), |m, b| {
                    *m = m.operate_with(b); // Reduction step.

                    // TODO: Should cleanup the buffer in the position of b
                    Some(m.clone())
                })
                .reduce(|g, m| g.operate_with(&m))
                .unwrap_or_else(Point::neutral_element)
        })
        .reduce(|t, g| t.operate_with_self(1_u64 << window_size).operate_with(&g))
        .ok_or_else(|| MetalError::LibraryError("placeholder_error".to_string()))
}

#[cfg(test)]
mod tests {
    use lambdaworks_math::{
        cyclic_group::IsGroup,
        elliptic_curve::{
            short_weierstrass::{
                curves::bls12_381::{curve::BLS12381Curve, field_extension::BLS12381FieldModulus},
                point::ShortWeierstrassProjectivePoint,
            },
            traits::{FromAffine, IsEllipticCurve},
        },
        field::{
            fields::montgomery_backed_prime_fields::MontgomeryBackendPrimeField, traits::IsField,
        },
        msm::pippenger,
        unsigned_integer::element::UnsignedInteger,
    };
    use proptest::{collection, prelude::*, prop_assert_eq, prop_compose, proptest};

    use crate::metal::abstractions::state::MetalState;

    const _CASES: u32 = 1;
    const _MAX_WSIZE: usize = 4;
    const _MAX_LEN: usize = 30;

    prop_compose! {
        fn unsigned_integer()(limbs: [u64; 6]) -> UnsignedInteger<6> {
            UnsignedInteger::from_limbs(limbs)
        }
    }

    prop_compose! {
        fn unsigned_integer_vec()(vec in collection::vec(unsigned_integer(), 0.._MAX_LEN)) -> Vec<UnsignedInteger<6>> {
            vec
        }
    }

    prop_compose! {
        fn point()(power: u128) -> <BLS12381Curve as IsEllipticCurve>::PointRepresentation {
            BLS12381Curve::generator().operate_with_self(power)
        }
    }

    prop_compose! {
        fn points_vec()(vec in collection::vec(point(), 0.._MAX_LEN)) -> Vec<<BLS12381Curve as IsEllipticCurve>::PointRepresentation> {
            vec
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: _CASES, .. ProptestConfig::default()
          })]
        // Property-based test that ensures the metal implementation matches the CPU one.
        #[test]
        fn test_metal_pippenger_matches_cpu(window_size in 1.._MAX_WSIZE, cs in unsigned_integer_vec(), hidings in points_vec()) {
            let state = MetalState::new(None).unwrap();
            let min_len = cs.len().min(hidings.len());
            let cs = cs[..min_len].to_vec();
            let hidings = hidings[..min_len].to_vec();

            let cpu_result = pippenger::msm_with(&cs, &hidings, window_size);
            let metal_result = super::pippenger(&cs, &hidings, window_size, &state).unwrap();

            prop_assert_eq!(metal_result, cpu_result);
        }
    }
}
