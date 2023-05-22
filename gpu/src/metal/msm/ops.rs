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

use metal::{ComputeCommandEncoderRef, MTLSize};

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

pub fn pippenger_sequencial<const NUM_LIMBS: usize>(
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

    let num_threads = 1;

    debug_assert!(window_size < usize::BITS);

    let pipeline = state.setup_pipeline("calculate_points_sum")?;

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

    let bucket_count = (1 << window_size) - 1;

    let mut results = vec![];
    let mut results_buffer = vec![];

    let result: Vec<u32> = vec![Point::neutral_element(); bucket_count]
        .into_iter()
        .map(|point| point.to_u32_limbs())
        .flatten()
        .collect();

    for _ in 0..num_windows {
        results.push(result.clone());
    }

    let cs_buffer = state.alloc_buffer_data(&cs);
    let hidings_buffer = state.alloc_buffer_data(&hidings);

    let threadgroup_size = MTLSize::new(1, 1, 1);
    let threadgroup_count = MTLSize::new(1, 1, 1);

    for window_idx in (0..num_windows as usize).rev() {
        let result_buffer = state.alloc_buffer_data(&results[window_idx]);
        results_buffer.push(result_buffer);

        let (command_buffer, command_encoder) = state.setup_command(
            &pipeline,
            Some(&[
                (0, &cs_buffer),
                (1, &hidings_buffer),
                (
                    4,
                    &results_buffer[(num_windows - window_idx as u64 - 1 as u64) as usize],
                ),
            ]),
        );

        command_encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            void_ptr(&(buflen as u32)),
        );
        command_encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            void_ptr(&(window_idx.clone() as u32)),
        );

        command_encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
        command_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed(); //TODO dispatch all and then wait
    }

    /*
        //TODO call other kernels
        let pipeline = state.setup_pipeline("calculate_window")?;

        let point_size = 12 * 3;

        let partial_sums: Vec<u32> = vec![Point::neutral_element(); bucket_count]
            .into_iter()
            .map(|point| point.to_u32_limbs())
            .flatten()
            .collect();

        let mut windows_buffer = vec![];
        for window_idx in 0..num_windows as usize {
            let partial_sums_buffer = state.alloc_buffer_data(&partial_sums.clone());
            let window_result_buffer = state.alloc_buffer::<u32>(point_size);
            windows_buffer.push(window_result_buffer);

            let (command_buffer, command_encoder) = state.setup_command(
                &pipeline,
                Some(&[
                    (0, &results_buffer[window_idx]),
                    (2, &partial_sums_buffer),
                    (3, &windows_buffer[window_idx]),
                ]),
            );

            command_encoder.set_bytes(
                1,
                std::mem::size_of::<u32>() as u64,
                void_ptr(&(bucket_count as u32)),
            );

            command_encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
            command_encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed(); //TODO dispatch all and then wait
        }

        let pipeline = state.setup_pipeline("reduce_windows")?;

        let mut windows_results = vec![];

        for window_buffer in windows_buffer {
            windows_results.push(MetalState::retrieve_contents::<u32>(&window_buffer));
        }

        let windows_results = windows_results.into_iter().flatten().collect::<Vec<u32>>();

        let reduce_windows_buffer = state.alloc_buffer_data(&windows_results);
        let reduced_buffer = state.alloc_buffer::<u32>(point_size);

        let (command_buffer, command_encoder) = state.setup_command(
            &pipeline,
            Some(&[(0, &reduce_windows_buffer), (2, &reduced_buffer)]),
        );

        command_encoder.set_bytes(
            1,
            std::mem::size_of::<u32>() as u64,
            void_ptr(&(num_windows as u32)),
        );

        command_encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
        command_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let result = MetalState::retrieve_contents::<u32>(&reduced_buffer);

        let result: Point = Point::from_u32_limbs(&result);

        Ok(result)
    */

    let mut buckets = vec![];

    for r in results_buffer {
        let bucket = MetalState::retrieve_contents::<u32>(&r);
        let points = bucket
            .chunks(12 * 3)
            .map(Point::from_u32_limbs)
            .collect::<Vec<Point>>();
        buckets.push(points);
    }

    let mut windows = vec![];
    for bucket in buckets {
        let mut partial_sum = Point::neutral_element();
        for (i, b) in bucket.iter().enumerate() {
            partial_sum = partial_sum.operate_with(&b.operate_with_self(i + 1));
        }
        windows.push(partial_sum);
    }

    let total = windows
        .into_iter()
        .reduce(|f, g| f.operate_with_self((1 << 4) as u64).operate_with(&g))
        .unwrap();

    Ok(total)
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

        #[test]
        fn test_metal_pippenger_sequencial_matches_cpu(window_size in 1.._MAX_WSIZE, cs in unsigned_integer_vec(), hidings in points_vec()) {
            let state = MetalState::new(None).unwrap();
            let min_len = cs.len().min(hidings.len());
            let cs = cs[..min_len].to_vec();
            let hidings = hidings[..min_len].to_vec();

            let cpu_result = pippenger::msm_with(&cs, &hidings, window_size);
            let metal_result = super::pippenger_sequencial(&cs, &hidings, &state).unwrap();

            prop_assert_eq!(metal_result, cpu_result);
        }
    }
}
