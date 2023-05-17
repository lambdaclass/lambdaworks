#[cfg(all(test, feature = "metal"))]
mod tests {
    use crate::metal::abstractions::state::MetalState;
    use lambdaworks_math::cyclic_group::IsGroup;
    use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381Curve;
    use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381PrimeField;
    use lambdaworks_math::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
    use lambdaworks_math::elliptic_curve::traits::{FromAffine, IsEllipticCurve};
    use lambdaworks_math::field::element::FieldElement;
    use lambdaworks_math::unsigned_integer::element::{UnsignedInteger, U384};
    use metal::MTLSize;
    use proptest::prelude::*;

    pub type F = BLS12381PrimeField;
    pub type FE = FieldElement<F>;
    pub type U = U384; // F::BaseType

    mod unsigned_int_tests {
        use super::*;

        enum BigOrSmallInt {
            Big(U),
            Small(usize),
        }

        fn execute_kernel(name: &str, params: (U, BigOrSmallInt)) -> U {
            let state = MetalState::new(None).unwrap();
            let pipeline = state.setup_pipeline(name).unwrap();

            let (a, b) = params;
            let a = a.to_u32_limbs();
            // conversion needed because of possible difference of endianess between host and
            // device (Metal's UnsignedInteger has 32bit limbs).

            let result_buffer = state.alloc_buffer::<U>(1);

            let (command_buffer, command_encoder) = match b {
                BigOrSmallInt::Big(b) => {
                    let b = b.to_u32_limbs();
                    let a_buffer = state.alloc_buffer_data(&a);
                    let b_buffer = state.alloc_buffer_data(&b);
                    state.setup_command(
                        &pipeline,
                        Some(&[(0, &a_buffer), (1, &b_buffer), (2, &result_buffer)]),
                    )
                }
                BigOrSmallInt::Small(b) => {
                    let a_buffer = state.alloc_buffer_data(&a);
                    let b_buffer = state.alloc_buffer_data(&[b]);
                    state.setup_command(
                        &pipeline,
                        Some(&[(0, &a_buffer), (1, &b_buffer), (2, &result_buffer)]),
                    )
                }
            };

            let threadgroup_size = MTLSize::new(1, 1, 1);
            let threadgroup_count = MTLSize::new(1, 1, 1);

            command_encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
            command_encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();

            let limbs = MetalState::retrieve_contents::<u32>(&result_buffer);
            U::from_u32_limbs(&limbs)
        }

        prop_compose! {
            fn rand_u()(n in any::<u128>()) -> U { U::from_u128(n) } // doesn't populate all limbs
        }

        use BigOrSmallInt::{Big, Small};

        proptest! {
            #[test]
            fn add(a in rand_u(), b in rand_u()) {
                objc::rc::autoreleasepool(|| {
                    let result = execute_kernel("test_uint_add", (a, Big(b)));
                    prop_assert_eq!(result, a + b);
                    Ok(())
                }).unwrap();
            }

            #[test]
            fn sub(a in rand_u(), b in rand_u()) {
                objc::rc::autoreleasepool(|| {
                    let a = std::cmp::max(a, b);
                    let b = std::cmp::min(a, b);

                    let result = execute_kernel("test_uint_sub", (a, Big(b)));
                    prop_assert_eq!(result, a - b);
                    Ok(())
                }).unwrap();
            }

            #[test]
            fn prod(a in rand_u(), b in rand_u()) {
                objc::rc::autoreleasepool(|| {
                    let result = execute_kernel("test_uint_prod", (a, Big(b)));
                    prop_assert_eq!(result, a * b);
                    Ok(())
                }).unwrap();
            }

            #[test]
            fn shl(a in rand_u(), b in any::<usize>()) {
                objc::rc::autoreleasepool(|| {
                    let b = b % 384; // so it doesn't overflow
                    let result = execute_kernel("test_uint_shl", (a, Small(b)));
                    prop_assert_eq!(result, a << b);
                    Ok(())
                }).unwrap();
            }

            #[test]
            fn shr(a in rand_u(), b in any::<usize>()) {
                objc::rc::autoreleasepool(|| {
                    let b = b % 384; // so it doesn't overflow
                    let result = execute_kernel("test_uint_shr", (a, Small(b)));
                    prop_assert_eq!(result, a >> b);
                    Ok(())
                }).unwrap();
            }
        }
    }

    mod fp_tests {
        use proptest::collection;

        use super::*;

        prop_compose! {
            fn rand_u32()(n in any::<u32>()) -> u32 { n }
        }

        prop_compose! {
            fn rand_limbs()(vec in collection::vec(rand_u32(), 12)) -> Vec<u32> {
                vec
            }
        }

        prop_compose! {
            fn rand_felt()(limbs in rand_limbs()) -> FE {
                FE::from(&U384::from_u32_limbs(&limbs))
            }
        }

        fn execute_kernel(name: &str, a: &FE, b: &FE) -> FE {
            let state = MetalState::new(None).unwrap();
            let pipeline = state.setup_pipeline(name).unwrap();

            // conversion needed because of possible difference of endianess between host and
            // device (Metal's UnsignedInteger has 32bit limbs).
            let a = a.value().to_u32_limbs();
            let b = b.value().to_u32_limbs();
            let a_buffer = state.alloc_buffer_data(&a);
            let b_buffer = state.alloc_buffer_data(&b);
            let result_buffer = state.alloc_buffer::<u32>(12);

            let (command_buffer, command_encoder) = state.setup_command(
                &pipeline,
                Some(&[(0, &a_buffer), (1, &b_buffer), (2, &result_buffer)]),
            );

            let threadgroup_size = MTLSize::new(1, 1, 1);
            let threadgroup_count = MTLSize::new(1, 1, 1);

            command_encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
            command_encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();

            let limbs = MetalState::retrieve_contents::<u32>(&result_buffer);
            FE::from_raw(&U::from_u32_limbs(&limbs))
        }

        proptest! {
            #[test]
            fn add(a in rand_felt(), b in rand_felt()) {
                objc::rc::autoreleasepool(|| {
                    let result = execute_kernel("fp_bls12381_add", &a, &b);
                    prop_assert_eq!(result, a + b);
                    Ok(())
                }).unwrap();
            }

            #[test]
            fn sub(a in rand_felt(), b in rand_felt()) {
                objc::rc::autoreleasepool(|| {
                    let result = execute_kernel("fp_bls12381_sub", &a, &b);
                    prop_assert_eq!(result, a - b);
                    Ok(())
                }).unwrap();
            }

            #[test]
            fn mul(a in rand_felt(), b in rand_felt()) {
                objc::rc::autoreleasepool(|| {
                    let result = execute_kernel("fp_bls12381_mul", &a, &b);
                    prop_assert_eq!(result, a * b);
                    Ok(())
                }).unwrap();
            }
        }
    }

    #[test]
    fn test_metal_ec_p_plus_infinity_should_equal_p() {
        let state = MetalState::new(None).unwrap();
        let pipeline = state.setup_pipeline("bls12381_add").unwrap();

        let px = FE::from(0);
        let py = FE::from(2);
        let p = BLS12381Curve::create_point_from_affine(px, py).unwrap();
        let p_coordinates: Vec<u32> = p
            .coordinates()
            .into_iter()
            .map(|felt| felt.value().to_u32_limbs())
            .flatten()
            .collect();

        let p_buffer = state.alloc_buffer_data(&p_coordinates);

        let q = BLS12381Curve::generator().operate_with_self(0_u64);
        let q_coordinates: Vec<u32> = q
            .coordinates()
            .into_iter()
            .map(|felt| felt.value().to_u32_limbs())
            .flatten()
            .collect();
        let q_buffer = state.alloc_buffer_data(&q_coordinates);

        let result_buffer = state.alloc_buffer::<u32>(36);

        let (command_buffer, command_encoder) = state.setup_command(
            &pipeline,
            Some(&[(0, &p_buffer), (1, &q_buffer), (2, &result_buffer)]),
        );

        let threadgroup_size = MTLSize::new(1, 1, 1);
        let threadgroup_count = MTLSize::new(1, 1, 1);
        command_encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
        command_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let result = MetalState::retrieve_contents::<u32>(&result_buffer);

        let cpu_result = p.operate_with(&q);
        let cpu_result_coordinates: Vec<u32> = cpu_result
            .coordinates()
            .into_iter()
            .map(|felt| felt.value().to_u32_limbs())
            .flatten()
            .collect();

        assert_eq!(cpu_result_coordinates, result);
    }

    #[test]
    fn test_metal_ec_infinity_plus_q_should_equal_q() {
        let state = MetalState::new(None).unwrap();
        let pipeline = state.setup_pipeline("bls12381_add").unwrap();

        let p = BLS12381Curve::generator().operate_with_self(0_u64);
        let p_coordinates: Vec<u32> = p
            .coordinates()
            .into_iter()
            .map(|felt| felt.value().to_u32_limbs())
            .flatten()
            .collect();
        let p_buffer = state.alloc_buffer_data(&p_coordinates);

        let qx = FE::from(0);
        let qy = FE::from(2);
        let q = BLS12381Curve::create_point_from_affine(qx, qy).unwrap();
        let q_coordinates: Vec<u32> = q
            .coordinates()
            .into_iter()
            .map(|felt| felt.value().to_u32_limbs())
            .flatten()
            .collect();

        let q_buffer = state.alloc_buffer_data(&q_coordinates);

        let result_buffer = state.alloc_buffer::<u32>(36);

        let (command_buffer, command_encoder) = state.setup_command(
            &pipeline,
            Some(&[(0, &p_buffer), (1, &q_buffer), (2, &result_buffer)]),
        );

        let threadgroup_size = MTLSize::new(1, 1, 1);
        let threadgroup_count = MTLSize::new(1, 1, 1);
        command_encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
        command_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let result = MetalState::retrieve_contents::<u32>(&result_buffer);

        let cpu_result = p.operate_with(&q);
        let cpu_result_coordinates: Vec<u32> = cpu_result
            .coordinates()
            .into_iter()
            .map(|felt| felt.value().to_u32_limbs())
            .flatten()
            .collect();

        assert_eq!(cpu_result_coordinates, result);
    }

    #[test]
    fn test_metal_p_plus_p_should_equal_cpu() {
        let state = MetalState::new(None).unwrap();
        let pipeline = state.setup_pipeline("bls12381_add").unwrap();

        let px = FE::from(0);
        let py = FE::from(2);
        let p = BLS12381Curve::create_point_from_affine(px, py).unwrap();
        let p_coordinates: Vec<u32> = p
            .coordinates()
            .into_iter()
            .map(|felt| felt.value().to_u32_limbs())
            .flatten()
            .collect();

        let p_buffer = state.alloc_buffer_data(&p_coordinates);

        let q = p.clone();
        let q_coordinates = p_coordinates.clone();

        let q_buffer = state.alloc_buffer_data(&q_coordinates);

        let result_buffer = state.alloc_buffer::<u32>(36);

        let (command_buffer, command_encoder) = state.setup_command(
            &pipeline,
            Some(&[(0, &p_buffer), (1, &q_buffer), (2, &result_buffer)]),
        );

        let threadgroup_size = MTLSize::new(1, 1, 1);
        let threadgroup_count = MTLSize::new(1, 1, 1);
        command_encoder.dispatch_thread_groups(threadgroup_count, threadgroup_size);
        command_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let result = MetalState::retrieve_contents::<u32>(&result_buffer);

        let cpu_result = p.operate_with(&q);
        let cpu_result_coordinates: Vec<u32> = cpu_result
            .coordinates()
            .into_iter()
            .map(|felt| felt.value().to_u32_limbs())
            .flatten()
            .collect();

        assert_eq!(result, cpu_result_coordinates);
    }
}
