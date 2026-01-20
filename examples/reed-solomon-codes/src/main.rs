//! Reed-Solomon Codes - Interactive Demonstrations
//!
//! This example demonstrates the core concepts of Reed-Solomon codes:
//! - Encoding messages as polynomial evaluations
//! - Unique decoding with Berlekamp-Welch
//! - List decoding with Sudan and Guruswami-Sudan algorithms
//! - Verifying that decoded candidates are within expected distance

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::polynomial::Polynomial;

use reed_solomon_codes::berlekamp_welch;
use reed_solomon_codes::distance::{hamming_distance, introduce_errors_at_positions};
use reed_solomon_codes::guruswami_sudan::{gs_decoding_radius, gs_list_decode};
use reed_solomon_codes::reed_solomon::ReedSolomonCode;
use reed_solomon_codes::sudan::{sudan_decoding_radius, sudan_list_decode};
use reed_solomon_codes::Babybear31PrimeField;

type FE = FieldElement<Babybear31PrimeField>;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                   Reed-Solomon Codes Demo                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    demo_encoding();
    demo_berlekamp_welch();
    demo_sudan_list_decoding();
    demo_guruswami_sudan_list_decoding();
}

/// Demonstrates basic RS encoding.
fn demo_encoding() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  1. REED-SOLOMON ENCODING");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let code = ReedSolomonCode::<Babybear31PrimeField>::new(16, 8);

    println!(
        "Code parameters: RS[{}, {}]",
        code.code_length(),
        code.dimension()
    );
    println!("  - Code length (n):        {}", code.code_length());
    println!("  - Dimension (k):          {}", code.dimension());
    println!("  - Minimum distance (d):   {}", code.minimum_distance());
    println!("  - Rate (k/n):             {:.2}", code.rate());
    println!(
        "  - Unique decoding radius: {} errors",
        code.unique_decoding_radius()
    );
    println!();

    let message: Vec<FE> = vec![
        FE::from(1u64),
        FE::from(2u64),
        FE::from(3u64),
        FE::from(4u64),
        FE::from(5u64),
        FE::from(6u64),
        FE::from(7u64),
        FE::from(8u64),
    ];

    println!(
        "Message (polynomial coefficients): {:?}",
        message
            .iter()
            .map(|x| x.representative())
            .collect::<Vec<_>>()
    );
    println!("  p(x) = 1 + 2x + 3x^2 + 4x^3 + 5x^4 + 6x^5 + 7x^6 + 8x^7");
    println!();

    let codeword = code.encode(&message);
    println!("Codeword (evaluations at {} points):", code.code_length());
    print!("  [");
    for (i, eval) in codeword.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{}", eval.representative());
    }
    println!("]");
    println!();
}

/// Demonstrates Berlekamp-Welch unique decoding with distance verification.
fn demo_berlekamp_welch() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  2. BERLEKAMP-WELCH UNIQUE DECODING");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let code = ReedSolomonCode::<Babybear31PrimeField>::with_consecutive_domain(16, 8);
    let t = code.unique_decoding_radius();

    println!("RS[16, 8] code over BabyBear field");
    println!("Unique decoding radius: t = floor((n-k)/2) = {} errors", t);
    println!();

    let message: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();
    let original_codeword = code.encode(&message);

    println!(
        "Original message: {:?}",
        message
            .iter()
            .map(|x| x.representative())
            .collect::<Vec<_>>()
    );
    println!();

    // Test decoding with errors up to and beyond the unique decoding radius
    for num_errors in [1, 2, 3, 4, 5] {
        let error_positions: Vec<usize> = (0..num_errors).map(|i| i * 3).collect();
        let received = introduce_errors_at_positions(&original_codeword, &error_positions);
        let actual_errors = hamming_distance(&original_codeword, &received);

        println!(
            "  Introducing {} errors at positions {:?}",
            actual_errors, error_positions
        );

        match berlekamp_welch::decode(&code, &received, None) {
            Ok(result) => {
                // Encode the recovered polynomial to get its codeword
                let recovered_coeffs: Vec<FE> = result.polynomial.coefficients().to_vec();
                let recovered_codeword = code.encode(&recovered_coeffs);

                // Verify distance from received word
                let dist_to_received = hamming_distance(&received, &recovered_codeword);
                let dist_to_original = hamming_distance(&original_codeword, &recovered_codeword);

                let recovered_msg: Vec<_> = recovered_coeffs
                    .iter()
                    .map(|x| x.representative())
                    .collect();

                println!("    Decoded polynomial: {:?}", recovered_msg);
                println!("    Distance to received word: {}", dist_to_received);
                println!("    Distance to original codeword: {}", dist_to_original);

                if dist_to_original == 0 {
                    println!("    => Correctly recovered original message");
                } else {
                    println!(
                        "    => Different from original (expected with {} > {} errors)",
                        actual_errors, t
                    );
                }
            }
            Err(e) => {
                println!("    Failed to decode: {}", e);
                if actual_errors > t {
                    println!(
                        "    => Expected failure: {} errors exceeds radius {}",
                        actual_errors, t
                    );
                }
            }
        }
        println!();
    }
}

/// Demonstrates Sudan's list decoding algorithm.
fn demo_sudan_list_decoding() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  3. SUDAN'S LIST DECODING");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Use RS[32, 4] which has better Sudan radius than BW radius
    let code = ReedSolomonCode::<Babybear31PrimeField>::with_consecutive_domain(32, 4);
    let bw_radius = code.unique_decoding_radius();
    let sudan_radius = sudan_decoding_radius(code.code_length(), code.dimension());

    println!("RS[32, 4] code over BabyBear field");
    println!("Berlekamp-Welch radius: {} errors", bw_radius);
    println!(
        "Sudan decoding radius:  {} errors (n - sqrt(2nk))",
        sudan_radius
    );
    println!();

    let message: Vec<FE> = (1..=4).map(|i| FE::from(i as u64)).collect();
    let original_codeword = code.encode(&message);
    let original_poly = Polynomial::new(&message);

    println!(
        "Original message: {:?}",
        message
            .iter()
            .map(|x| x.representative())
            .collect::<Vec<_>>()
    );
    println!();

    // Test with various error counts
    for num_errors in [0, 5, 10, 12] {
        let error_positions: Vec<usize> = (0..num_errors).collect();
        let received = introduce_errors_at_positions(&original_codeword, &error_positions);
        let actual_errors = hamming_distance(&original_codeword, &received);

        if actual_errors == 0 {
            println!("  No errors:");
        } else {
            println!("  {} errors at positions 0..{}:", actual_errors, num_errors);
        }

        let result = sudan_list_decode(&code, &received);

        println!("    List size: {}", result.candidates.len());

        // Show all candidates with their distances
        let max_display = 10;
        for (i, candidate) in result.candidates.iter().take(max_display).enumerate() {
            let coeffs: Vec<FE> = candidate.coefficients().to_vec();
            let candidate_codeword = code.encode(&coeffs);
            let dist_to_received = hamming_distance(&received, &candidate_codeword);
            let dist_to_original = hamming_distance(&original_codeword, &candidate_codeword);

            let coeffs_repr: Vec<_> = coeffs.iter().map(|x| x.representative()).collect();

            let is_original = candidate == &original_poly;
            let marker = if is_original { " <-- ORIGINAL" } else { "" };

            println!("      [{}] {:?}{}", i + 1, coeffs_repr, marker);
            println!(
                "          dist to received: {}, dist to original: {}",
                dist_to_received, dist_to_original
            );
        }
        if result.candidates.len() > max_display {
            println!(
                "      ... and {} more candidates",
                result.candidates.len() - max_display
            );
        }

        let found = result.candidates.contains(&original_poly);
        println!("    Original in list: {}", if found { "YES" } else { "NO" });
        println!();
    }
}

/// Demonstrates Guruswami-Sudan list decoding algorithm.
fn demo_guruswami_sudan_list_decoding() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  4. GURUSWAMI-SUDAN LIST DECODING");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    // Use RS[16, 4] which has a good ratio for demonstrating list decoding
    let code = ReedSolomonCode::<Babybear31PrimeField>::with_consecutive_domain(16, 4);
    let bw_radius = code.unique_decoding_radius();
    let sudan_radius = sudan_decoding_radius(code.code_length(), code.dimension());
    let gs_radius = gs_decoding_radius(code.code_length(), code.dimension());

    println!("RS[16, 4] code over BabyBear field");
    println!("Berlekamp-Welch radius: {} errors", bw_radius);
    println!(
        "Sudan radius:           {} errors (n - sqrt(2nk))",
        sudan_radius
    );
    println!(
        "Guruswami-Sudan radius: {} errors (n - sqrt(nk))",
        gs_radius
    );
    println!();

    let message: Vec<FE> = (1..=4).map(|i| FE::from(i as u64)).collect();
    let original_codeword = code.encode(&message);
    let original_poly = Polynomial::new(&message);

    println!(
        "Original message: {:?}",
        message
            .iter()
            .map(|x| x.representative())
            .collect::<Vec<_>>()
    );
    println!();

    // Test with various error counts up to and beyond GS radius
    for num_errors in [0, 2, 4, 6, 7] {
        let error_positions: Vec<usize> = (0..num_errors).collect();
        let received = introduce_errors_at_positions(&original_codeword, &error_positions);
        let actual_errors = hamming_distance(&original_codeword, &received);

        if actual_errors == 0 {
            println!("  No errors:");
        } else {
            println!("  {} errors at positions 0..{}:", actual_errors, num_errors);
        }

        let result = gs_list_decode(&code, &received);

        println!(
            "    List size: {} (multiplicity={})",
            result.candidates.len(),
            result.multiplicity
        );

        // Show all candidates with their distances
        let max_display = 10;
        for (i, candidate) in result.candidates.iter().take(max_display).enumerate() {
            let coeffs: Vec<FE> = candidate.coefficients().to_vec();
            let candidate_codeword = code.encode(&coeffs);
            let dist_to_received = hamming_distance(&received, &candidate_codeword);
            let dist_to_original = hamming_distance(&original_codeword, &candidate_codeword);

            let coeffs_repr: Vec<_> = coeffs.iter().map(|x| x.representative()).collect();

            let is_original = candidate == &original_poly;
            let marker = if is_original { " <-- ORIGINAL" } else { "" };

            println!("      [{}] {:?}{}", i + 1, coeffs_repr, marker);
            println!(
                "          dist to received: {}, dist to original: {}",
                dist_to_received, dist_to_original
            );
        }
        if result.candidates.len() > max_display {
            println!(
                "      ... and {} more candidates",
                result.candidates.len() - max_display
            );
        }

        let found = result.candidates.contains(&original_poly);
        println!(
            "    Original in list: {}{}",
            if found { "YES" } else { "NO" },
            if !found && actual_errors > gs_radius {
                format!(
                    " (expected: {} errors > radius {})",
                    actual_errors, gs_radius
                )
            } else {
                String::new()
            }
        );
        println!();
    }

    // Demo with RS[32, 4] - longer code, same dimension
    println!("  --- RS[32, 4] Demo (rate = 0.125) ---");
    println!();

    let code3 = ReedSolomonCode::<Babybear31PrimeField>::with_consecutive_domain(32, 4);
    let gs_radius3 = gs_decoding_radius(32, 4);
    let bw_radius3 = code3.unique_decoding_radius();
    let sudan_radius3 = sudan_decoding_radius(32, 4);

    println!(
        "  RS[32, 4] code: BW = {}, Sudan = {}, GS = {} errors",
        bw_radius3, sudan_radius3, gs_radius3
    );

    let msg3: Vec<FE> = (1..=4).map(|i| FE::from(i as u64)).collect();
    let cw3 = code3.encode(&msg3);
    let original_poly3 = Polynomial::new(&msg3);

    println!(
        "  Original message: {:?}",
        msg3.iter().map(|x| x.representative()).collect::<Vec<_>>()
    );
    println!();

    for num_errors in [10, 15, 18, 20] {
        let error_positions: Vec<usize> = (0..num_errors).collect();
        let received3 = introduce_errors_at_positions(&cw3, &error_positions);

        println!("  {} errors:", num_errors);

        let result3 = gs_list_decode(&code3, &received3);
        println!(
            "    List size: {} (multiplicity={})",
            result3.candidates.len(),
            result3.multiplicity
        );

        for (i, candidate) in result3.candidates.iter().take(5).enumerate() {
            let coeffs: Vec<FE> = candidate.coefficients().to_vec();
            let candidate_codeword = code3.encode(&coeffs);
            let dist_to_received = hamming_distance(&received3, &candidate_codeword);
            let dist_to_original = hamming_distance(&cw3, &candidate_codeword);

            let coeffs_repr: Vec<_> = coeffs.iter().map(|x| x.representative()).collect();

            let is_original = candidate == &original_poly3;
            let marker = if is_original { " <-- ORIGINAL" } else { "" };

            println!("      [{}] {:?}{}", i + 1, coeffs_repr, marker);
            println!(
                "          dist to received: {}, dist to original: {}",
                dist_to_received, dist_to_original
            );
        }
        if result3.candidates.len() > 5 {
            println!("      ... and {} more", result3.candidates.len() - 5);
        }

        let found = result3.candidates.contains(&original_poly3);
        println!("    Original in list: {}", if found { "YES" } else { "NO" });
        println!();
    }

    // Comparison summary
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  DECODING RADIUS COMPARISON");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();
    println!("  ┌──────────┬─────────────────┬─────────────────┬─────────────────┐");
    println!("  │   Code   │ Berlekamp-Welch │     Sudan       │ Guruswami-Sudan │");
    println!("  │          │   (n-k)/2       │  n - sqrt(2nk)  │   n - sqrt(nk)  │");
    println!("  ├──────────┼─────────────────┼─────────────────┼─────────────────┤");

    for (n, k) in [(16, 4), (32, 8), (64, 16), (32, 4)] {
        let bw = (n - k) / 2;
        let sudan = sudan_decoding_radius(n, k);
        let gs = gs_decoding_radius(n, k);

        println!(
            "  │ RS[{:2},{:2}] │       {:3}       │       {:3}       │       {:3}       │",
            n, k, bw, sudan, gs
        );
    }
    println!("  └──────────┴─────────────────┴─────────────────┴─────────────────┘");
    println!();
}
