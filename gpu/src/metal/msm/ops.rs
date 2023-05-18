#![allow(unused)]
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::short_weierstrass::{
        curves::bls12_381::{curve::BLS12381Curve, field_extension::BLS12381PrimeField},
        point::ShortWeierstrassProjectivePoint,
    },
    field::{element::FieldElement, traits::IsField},
    unsigned_integer::element::UnsignedInteger,
};

use crate::metal::{
    abstractions::{errors::MetalError, state::*},
    helpers::void_ptr,
};

use metal::MTLSize;

use core::mem;

type Point = ShortWeierstrassProjectivePoint<BLS12381Curve>;
pub type F = BLS12381PrimeField;
pub type FE = FieldElement<F>;

/// Executes parallel ordered FFT over a slice of two-adic field elements, in Metal.
/// Twiddle factors are required to be in bit-reverse order.
///
/// "Ordered" means that the input is required to be in natural order, and the output will be
/// in this order too. Natural order means that input[i] corresponds to the i-th coefficient,
/// as opposed to bit-reverse order in which input[bit_rev(i)] corresponds to the i-th
/// coefficient.
// TODO: to support big endian architecture, copy all limbs with indices changed: 103254 -> 012345
#[cfg(target_endian = "little")]
pub fn pippenger<const NUM_LIMBS: usize>(
    cs: &[UnsignedInteger<NUM_LIMBS>],
    hidings: &[Point],
    state: &MetalState,
) -> Result<Point, MetalError> {
    use lambdaworks_math::{
        elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381PrimeField,
        unsigned_integer::{element::U384, traits::U32Limbs},
    };

    debug_assert_eq!(
        cs.len(),
        hidings.len(),
        "Slices `cs` and `hidings` must be of the same length to compute `msm`."
    );

    const MAX_THREADS: u64 = 64;

    let window_size: u32 = 4;
    let buflen: u64 = cs.len() as u64;
    let n_bits = 64 * NUM_LIMBS as u64;

    let num_windows = (n_bits - 1) / window_size as u64 + 1;

    let num_threads = MAX_THREADS.min(buflen);

    debug_assert!(window_size < usize::BITS);

    let pipeline = state.setup_pipeline("calculate_Gjs_bls12381")?;

    let cs: Vec<u32> = cs
        .into_iter()
        .map(|uint| uint.to_u32_limbs())
        .flatten()
        .collect();

    let hidings: Vec<u32> = hidings
        .into_iter()
        .map(|point| point.to_u32_limbs())
        .flatten()
        .collect();

    let cs_buffer = state.alloc_buffer_data(&cs);
    let hidings_buffer = state.alloc_buffer_data(&hidings);
    let result_buffer = state.alloc_buffer::<u32>(num_windows as usize * 3 * 12);

    let (command_buffer, command_encoder) = state.setup_command(
        &pipeline,
        Some(&[(0, &cs_buffer), (1, &hidings_buffer), (4, &result_buffer)]),
    );

    command_encoder.set_bytes(2, std::mem::size_of::<u32>() as u64, void_ptr(&window_size));
    command_encoder.set_bytes(3, std::mem::size_of::<u64>() as u64, void_ptr(&buflen));

    let threadgroup_size = MTLSize::new(num_threads, 1, 1);
    let threadgroup_count = MTLSize::new(num_windows, 1, 1);

    command_encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
    command_encoder.end_encoding();

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let result: Vec<u32> = MetalState::retrieve_contents(&result_buffer);

    let result: Vec<Point> = result.chunks(12 * 3).map(Point::from_u32_limbs).collect();

    // TODO: do this in GPU
    let result = result
        .into_iter()
        .rev()
        .reduce(|t, g| t.operate_with_self(1_u64 << window_size).operate_with(&g));

    Ok(result.expect("result_buffer is never empty"))
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
            let metal_result = super::pippenger(&cs, &hidings, &state).unwrap();

            prop_assert_eq!(metal_result, cpu_result);
        }
    }
}
