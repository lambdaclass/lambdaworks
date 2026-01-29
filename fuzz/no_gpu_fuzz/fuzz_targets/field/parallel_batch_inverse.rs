#![no_main]
//! Fuzz tests for parallel batch inversion optimization.
//!
//! Tests that the parallel batch inversion (using chunked Montgomery's trick)
//! produces identical results to sequential batch inversion.
//!
//! Optimization: Uses rayon to parallelize batch inversion for large arrays (>= 4096 elements).
//! Each chunk performs independent batch inversion, trading one inversion per chunk
//! for parallelism speedup.

use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381PrimeField,
    field::element::FieldElement,
};
use libfuzzer_sys::fuzz_target;

type Fp = FieldElement<BLS12381PrimeField>;

fuzz_target!(|data: Vec<u64>| {
    // Limit size to avoid excessive runtime
    if data.is_empty() || data.len() > 10000 {
        return;
    }

    // Convert u64 inputs to field elements, filtering out zeros
    let mut elements: Vec<Fp> = data
        .iter()
        .map(|&x| Fp::from(x))
        .filter(|x| x != &Fp::zero())
        .collect();

    if elements.is_empty() {
        return;
    }

    // Clone for parallel test
    let mut elements_parallel = elements.clone();

    // ===== TEST 1: Parallel vs sequential consistency =====
    #[cfg(feature = "parallel")]
    {
        let seq_result = Fp::inplace_batch_inverse(&mut elements);
        let par_result = Fp::inplace_batch_inverse_parallel(&mut elements_parallel);

        // Both should succeed or both should fail
        match (seq_result, par_result) {
            (Ok(()), Ok(())) => {
                // Compare results element by element
                assert_eq!(
                    elements.len(),
                    elements_parallel.len(),
                    "Length mismatch after batch inverse"
                );

                for (i, (seq, par)) in elements.iter().zip(elements_parallel.iter()).enumerate() {
                    assert_eq!(
                        seq, par,
                        "Mismatch at index {}: sequential={:?}, parallel={:?}",
                        i, seq, par
                    );
                }
            }
            (Err(_), Err(_)) => {
                // Both failed, which is acceptable
            }
            (Ok(()), Err(e)) => {
                panic!("Sequential succeeded but parallel failed: {:?}", e);
            }
            (Err(e), Ok(())) => {
                panic!("Parallel succeeded but sequential failed: {:?}", e);
            }
        }
    }

    // ===== TEST 2: Verify inversion property =====
    // For all inverted elements: x * x^(-1) = 1
    let original_elements: Vec<Fp> = data
        .iter()
        .map(|&x| Fp::from(x))
        .filter(|x| x != &Fp::zero())
        .collect();

    if Fp::inplace_batch_inverse(&mut elements).is_ok() {
        for (orig, inv) in original_elements.iter().zip(elements.iter()) {
            let product = orig * inv;
            assert_eq!(
                product,
                Fp::one(),
                "Inversion property failed: x * x^(-1) != 1"
            );
        }
    }

    // ===== TEST 3: Edge case - single element =====
    if !original_elements.is_empty() {
        let mut single = vec![original_elements[0].clone()];
        let single_result = Fp::inplace_batch_inverse(&mut single);
        assert!(single_result.is_ok(), "Single element batch inverse failed");

        let product = &original_elements[0] * &single[0];
        assert_eq!(product, Fp::one(), "Single element inversion incorrect");
    }

    // ===== TEST 4: Edge case - all same element =====
    if !original_elements.is_empty() && data.len() >= 3 {
        let same_elem = original_elements[0].clone();
        let mut all_same = vec![same_elem.clone(); data.len().min(100)];
        let original_same = all_same.clone();

        if Fp::inplace_batch_inverse(&mut all_same).is_ok() {
            // All inverses should be equal
            for inv in &all_same {
                let product = &same_elem * inv;
                assert_eq!(product, Fp::one(), "All-same inversion property failed");
            }

            // Verify all inverted values are identical
            for i in 1..all_same.len() {
                assert_eq!(
                    all_same[0], all_same[i],
                    "Inverses of identical elements should be identical"
                );
            }
        }
    }

    // ===== TEST 5: Large batch (triggers parallel path) =====
    #[cfg(feature = "parallel")]
    if data.len() >= 10 {
        // Create a larger batch by repeating the data
        let large_size = 4096; // Threshold for parallel execution
        let mut large_batch: Vec<Fp> = Vec::with_capacity(large_size);
        for i in 0..large_size {
            let val = Fp::from(data[i % data.len()]);
            if !val.is_zero() {
                large_batch.push(val);
            }
        }

        if !large_batch.is_empty() {
            let large_batch_original = large_batch.clone();
            let mut large_batch_seq = large_batch.clone();
            let mut large_batch_par = large_batch.clone();

            let seq_result = Fp::inplace_batch_inverse(&mut large_batch_seq);
            let par_result = Fp::inplace_batch_inverse_parallel(&mut large_batch_par);

            if seq_result.is_ok() && par_result.is_ok() {
                // Verify they match
                for (i, (seq, par)) in large_batch_seq
                    .iter()
                    .zip(large_batch_par.iter())
                    .enumerate()
                {
                    assert_eq!(seq, par, "Large batch mismatch at index {}", i);
                }

                // Verify inversion property
                for (orig, inv) in large_batch_original.iter().zip(large_batch_par.iter()) {
                    let product = orig * inv;
                    assert_eq!(product, Fp::one(), "Large batch inversion property failed");
                }
            }
        }
    }

    // ===== TEST 6: Idempotence - double inversion =====
    // (x^(-1))^(-1) = x
    if !original_elements.is_empty() && original_elements.len() <= 100 {
        let mut once_inverted = original_elements.clone();
        if Fp::inplace_batch_inverse(&mut once_inverted).is_ok() {
            let mut twice_inverted = once_inverted.clone();
            if Fp::inplace_batch_inverse(&mut twice_inverted).is_ok() {
                for (orig, double_inv) in original_elements.iter().zip(twice_inverted.iter()) {
                    assert_eq!(orig, double_inv, "Double inversion should return original");
                }
            }
        }
    }
});
