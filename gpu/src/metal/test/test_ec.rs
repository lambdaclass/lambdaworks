#[cfg(all(test, feature = "metal"))]
mod test {
    use crate::metal::abstractions::state::MetalState;
    use lambdaworks_math::field::element::FieldElement;
    use metal::MTLSize;

    pub type F = BLS12381PrimeField;
    use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381PrimeField;
    pub type FE = FieldElement<F>;

    #[test]
    fn test_metal_ec_p_plus_infinity_should_equal_p() {
        let state = MetalState::new(None).unwrap();
        let pipeline = state.setup_pipeline("bls12381_add").unwrap();

        let p: &[FE] = &[FE::from(2_u64), FE::from(1_u64), FE::from(3_u64)];
        let p_buffer = state.alloc_buffer_data(p);
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

        assert_eq!(result[0], FE::from(2_u64));
        assert_eq!(result[1], FE::from(1_u64));
        assert_eq!(result[2], FE::from(3_u64));
    }

    #[test]
    fn test_metal_ec_infinity_plus_q_should_equal_q() {
        let state = MetalState::new(None).unwrap();
        let pipeline = state.setup_pipeline("bls12381_add").unwrap();

        let p: &[FE] = &[FE::zero(), FE::one(), FE::from(179_u64)];
        let p_buffer = state.alloc_buffer_data(p);
        let q: &[FE] = &[FE::from(66_u64), FE::from(85_u64), FE::from(179_u64)];
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

        assert_eq!(result[0], FE::from(66_u64));
        assert_eq!(result[1], FE::from(85_u64));
        assert_eq!(result[2], FE::from(179_u64));
    }
}
