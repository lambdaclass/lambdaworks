#[cfg(all(test, feature = "metal"))]
mod test {
    use crate::metal::abstractions::state::MetalState;
    use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::curve::BLS12381Curve;
    use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381PrimeField;
    use lambdaworks_math::elliptic_curve::short_weierstrass::point::ShortWeierstrassProjectivePoint;
    use lambdaworks_math::elliptic_curve::traits::FromAffine;
    use lambdaworks_math::field::element::FieldElement;
    use lambdaworks_math::unsigned_integer::element::{UnsignedInteger, U384};
    use metal::MTLSize;

    pub type F = BLS12381PrimeField;
    pub type FE = FieldElement<F>;

    #[test]
    fn test_metal_add_fp_should_equal_cpu() {
        objc::rc::autoreleasepool(|| {
            let state = MetalState::new(None).unwrap();
            let pipeline = state.setup_pipeline("fp_bls12381_add").unwrap();

            let p = FE::from(555);
            let p_buffer = state.alloc_buffer_data(&[p.clone()]);

            let q = FE::from(666);
            let q_buffer = state.alloc_buffer_data(&[q.clone()]);
            let result_buffer = state.alloc_buffer_data(&[FE::zero()]);

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

            let result: Vec<FE> = result.iter().map(FieldElement::from_raw).collect();

            assert_eq!(result[0], p + q);
        });
    }

    #[test]
    fn test_metal_mul_fp_should_equal_cpu() {
        let state = MetalState::new(None).unwrap();
        let pipeline = state.setup_pipeline("fp_bls12381_mul").unwrap();

        let p = FE::from(555);
        let p_buffer = state.alloc_buffer_data(&[p.clone()]);

        let q = FE::from(666);
        let q_buffer = state.alloc_buffer_data(&[q.clone()]);
        let result_buffer = state.alloc_buffer_data(&[FE::zero()]);

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

        let result: Vec<FE> = MetalState::retrieve_contents(&result_buffer);

        assert_eq!(result[0], p * q);
    }

    #[test]
    fn test_metal_ec_p_plus_infinity_should_equal_p() {
        let state = MetalState::new(None).unwrap();
        let pipeline = state.setup_pipeline("bls12381_add").unwrap();

        let px = FE::from(0);
        let py = FE::from(2);
        let p = ShortWeierstrassProjectivePoint::<BLS12381Curve>::from_affine(px, py).unwrap();

        let p_buffer = state.alloc_buffer_data(p.coordinates());
        let q: &[FE] = &[FE::zero(), FE::one(), FE::zero()];
        let q_buffer = state.alloc_buffer_data(q);
        let result: &[FE] = &[FE::from(1), FE::from(1), FE::from(1)];
        let result_buffer = state.alloc_buffer_data(result);

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

        let result: Vec<FE> = MetalState::retrieve_contents(&result_buffer);

        assert_eq!(&result[0], p.x());
        assert_eq!(&result[1], p.y());
        assert_eq!(&result[2], p.z());
    }

    #[test]
    fn test_metal_ec_infinity_plus_q_should_equal_q() {
        let state = MetalState::new(None).unwrap();
        let pipeline = state.setup_pipeline("bls12381_add").unwrap();

        let p: &[FE] = &[FE::zero(), FE::one(), FE::zero()];
        let p_buffer = state.alloc_buffer_data(p);

        let qx = FE::from(0);
        let qy = FE::from(2);
        let q = ShortWeierstrassProjectivePoint::<BLS12381Curve>::from_affine(qx, qy).unwrap();

        let q_buffer = state.alloc_buffer_data(q.coordinates());
        let result: &[FE] = &[FE::from(1), FE::from(1), FE::from(1)];
        let result_buffer = state.alloc_buffer_data(result);

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

        let result: Vec<FE> = MetalState::retrieve_contents(&result_buffer);

        assert_eq!(&result[0], q.x());
        assert_eq!(&result[1], q.y());
        assert_eq!(&result[2], q.z());
    }

    #[test]
    fn test_metal_sum_should_equal_cpu_sum() {
        objc::rc::autoreleasepool(|| {
            let state = MetalState::new(None).unwrap();
            let pipeline = state.setup_pipeline("bls12381_add").unwrap();

            // let px = FE::from(0);
            // let py = FE::from(2);
            // let p = ShortWeierstrassProjectivePoint::<BLS12381Curve>::from_affine(px, py).unwrap();

            let qx = FE::from(0);
            let qy = FE::from(2);
            let q = ShortWeierstrassProjectivePoint::<BLS12381Curve>::from_affine(qx, qy).unwrap();

            let p =FE::from_hex("44ed1dea66f6a8b9d8561ee71e58fa8339819b906d8dd6b77ce7fb8812cd333c4470d257e33bed03a8dfbb765f8a5c6");
            let p_buffer = state.alloc_buffer_data(&[FE::one()]);
            let q_buffer = state.alloc_buffer_data(q.coordinates());
            let result_buffer = state.alloc_buffer::<FE>(1);

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

            let result: Vec<UnsignedInteger<6>> = MetalState::retrieve_contents(&result_buffer);
            let result: Vec<FE> = result.iter().map(FE::from).collect();
            // let result_cpu = p.operate_with(&q);

            eprintln!("{:x?}", result[0]);
            eprintln!("{:x?}", FE::one());
            eprintln!("{:x?}", FE::from_raw(&FE::one().representative()));
            let val = result[0].clone();
            // let val: Vec<_> = val
            //     .representative()
            //     .limbs
            //     .into_iter()
            //     .map(|v| ((v & 0xFFFF) << 32) | (v >> 32))
            //     .collect();
            // let val: [u64; 6] = val.try_into().unwrap();
            // let val = FE::from_raw(&UnsignedInteger::<6>::from_limbs(val));
            assert_eq!(val, FE::one());

            //assert_eq!(&result[0], result_cpu.x());
            //assert_eq!(&result[1], result_cpu.y());
            //assert_eq!(&result[2], result_cpu.z());
        });
    }
}
