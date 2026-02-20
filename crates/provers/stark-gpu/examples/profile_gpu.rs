/// Detailed profiling of GPU STARK prover phases and sub-operations.
///
/// Run with: cargo run -p lambdaworks-stark-gpu --features metal --example profile_gpu --release
use std::time::Instant;

use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use lambdaworks_crypto::fiat_shamir::is_transcript::IsTranscript;
use lambdaworks_math::field::{
    element::FieldElement, fields::u64_goldilocks_field::Goldilocks64Field,
};
use lambdaworks_stark_gpu::metal::prover::prove_gpu_optimized;
use stark_platinum_prover::{
    examples::fibonacci_rap::{fibonacci_rap_trace, FibonacciRAP, FibonacciRAPPublicInputs},
    proof::options::ProofOptions,
    prover::{IsStarkProver, Prover},
    traits::AIR,
};

type F = Goldilocks64Field;
type FpE = FieldElement<F>;

fn main() {
    let log_len: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);

    let trace_length: usize = 1 << log_len;
    println!("=== Profiling GPU STARK prover at 2^{log_len} ({trace_length} rows) ===\n");

    let pub_inputs = FibonacciRAPPublicInputs {
        steps: trace_length,
        a0: FpE::one(),
        a1: FpE::one(),
    };
    let proof_options = ProofOptions::default_test_options();

    // Warmup: compile shaders once
    {
        let mut trace = fibonacci_rap_trace::<F>([FpE::one(), FpE::one()], 16);
        let air = FibonacciRAP::new(
            trace.num_rows(),
            &FibonacciRAPPublicInputs {
                steps: 16,
                a0: FpE::one(),
                a1: FpE::one(),
            },
            &proof_options,
        );
        let mut transcript = DefaultTranscript::<F>::new(&[]);
        let _ = prove_gpu_optimized(&air, &mut trace, &mut transcript);
    }

    // CPU reference
    let start = Instant::now();
    let mut cpu_trace = fibonacci_rap_trace::<F>([FpE::one(), FpE::one()], trace_length);
    let air = FibonacciRAP::new(cpu_trace.num_rows(), &pub_inputs, &proof_options);
    let mut cpu_transcript = DefaultTranscript::<F>::new(&[]);
    let _ = Prover::<F, F, _>::prove(&air, &mut cpu_trace, &mut cpu_transcript).unwrap();
    let cpu_total = start.elapsed();
    println!("CPU total:         {:>10.2?}", cpu_total);

    // GPU optimized (total)
    let start = Instant::now();
    let mut gpu_trace = fibonacci_rap_trace::<F>([FpE::one(), FpE::one()], trace_length);
    let air = FibonacciRAP::new(gpu_trace.num_rows(), &pub_inputs, &proof_options);
    let mut gpu_transcript = DefaultTranscript::<F>::new(&[]);
    let _ = prove_gpu_optimized(&air, &mut gpu_trace, &mut gpu_transcript).unwrap();
    let gpu_total = start.elapsed();
    println!(
        "GPU_opt total:     {:>10.2?}  ({:.2}x vs CPU)",
        gpu_total,
        gpu_total.as_secs_f64() / cpu_total.as_secs_f64()
    );

    // Now run the instrumented version
    println!("\n--- Phase breakdown (GPU_opt) ---\n");
    profile_gpu_optimized(trace_length, &pub_inputs, &proof_options);
}

fn profile_gpu_optimized(
    trace_length: usize,
    pub_inputs: &FibonacciRAPPublicInputs<F>,
    proof_options: &ProofOptions,
) {
    use lambdaworks_stark_gpu::metal::constraint_eval::FibRapConstraintState;
    use lambdaworks_stark_gpu::metal::deep_composition::{
        DeepCompositionState, DomainInversionState,
    };
    use lambdaworks_stark_gpu::metal::fft::CosetShiftState;
    use lambdaworks_stark_gpu::metal::merkle::GpuKeccakMerkleState;
    use lambdaworks_stark_gpu::metal::phases::composition::gpu_round_2_goldilocks_merkle;
    use lambdaworks_stark_gpu::metal::phases::fri::{gpu_round_4_goldilocks, FriFoldState};
    use lambdaworks_stark_gpu::metal::phases::ood::gpu_round_3;
    use lambdaworks_stark_gpu::metal::phases::rap::gpu_round_1_goldilocks;
    use lambdaworks_stark_gpu::metal::state::StarkMetalState;
    use stark_platinum_prover::domain::Domain;

    let mut trace = fibonacci_rap_trace::<F>([FpE::one(), FpE::one()], trace_length);
    let air = FibonacciRAP::new(trace.num_rows(), pub_inputs, proof_options);
    let mut transcript = DefaultTranscript::<F>::new(&[]);

    // Shader compilation
    let t = Instant::now();
    let state = StarkMetalState::new().unwrap();
    let constraint_state = FibRapConstraintState::new().unwrap();
    let deep_comp_state = DeepCompositionState::new().unwrap();
    let keccak_state = GpuKeccakMerkleState::new().unwrap();
    let coset_state = CosetShiftState::new().unwrap();
    let fri_fold_state = FriFoldState::new().unwrap();
    let domain_inv_state = DomainInversionState::new().unwrap();
    let domain = Domain::new(&air);
    println!("  Setup (shaders):   {:>10.2?}", t.elapsed());

    // Phase 1: RAP
    let t = Instant::now();
    let round_1 = gpu_round_1_goldilocks(
        &air,
        &mut trace,
        &domain,
        &mut transcript,
        &state,
        &keccak_state,
    )
    .unwrap();
    let phase1_time = t.elapsed();
    println!("  Phase 1 (RAP):     {:>10.2?}", phase1_time);

    // Phase 2: Composition
    let t = Instant::now();
    let round_2 = gpu_round_2_goldilocks_merkle(
        &air,
        &domain,
        &round_1,
        &mut transcript,
        &state,
        Some(&constraint_state),
        &keccak_state,
        &coset_state,
    )
    .unwrap();
    let phase2_time = t.elapsed();
    println!("  Phase 2 (Comp):    {:>10.2?}", phase2_time);

    // Phase 3: OOD
    let t = Instant::now();
    let round_3 = gpu_round_3(&air, &domain, &round_1, &round_2, &mut transcript).unwrap();
    let phase3_time = t.elapsed();
    println!("  Phase 3 (OOD):     {:>10.2?}", phase3_time);

    // Phase 4: FRI
    let t = Instant::now();
    let _round_4 = gpu_round_4_goldilocks(
        &air,
        &domain,
        &round_1,
        &round_2,
        &round_3,
        &mut transcript,
        &state,
        Some(&deep_comp_state),
        &keccak_state,
        &coset_state,
        &fri_fold_state,
        Some(&domain_inv_state),
    )
    .unwrap();
    let phase4_time = t.elapsed();
    println!("  Phase 4 (FRI):     {:>10.2?}", phase4_time);

    let total = phase1_time + phase2_time + phase3_time + phase4_time;
    println!("\n  Phases total:      {:>10.2?}", total);

    // Now profile sub-operations in Phase 1
    println!("\n--- Phase 1 sub-operations ---\n");
    profile_phase1(trace_length, pub_inputs, proof_options);

    // Profile sub-operations in Phase 2
    println!("\n--- Phase 2 sub-operations ---\n");
    profile_phase2(trace_length, pub_inputs, proof_options);

    // Profile sub-operations in Phase 4
    println!("\n--- Phase 4 sub-operations ---\n");
    profile_phase4(trace_length, pub_inputs, proof_options);
}

fn profile_phase1(
    trace_length: usize,
    pub_inputs: &FibonacciRAPPublicInputs<F>,
    proof_options: &ProofOptions,
) {
    use lambdaworks_math::polynomial::Polynomial;
    use lambdaworks_stark_gpu::metal::fft::{gpu_evaluate_offset_fft, gpu_interpolate_fft};
    use lambdaworks_stark_gpu::metal::merkle::{
        gpu_build_merkle_tree, gpu_hash_leaves_goldilocks, gpu_transpose_bitrev,
        GpuKeccakMerkleState,
    };
    use lambdaworks_stark_gpu::metal::state::StarkMetalState;
    use stark_platinum_prover::trace::columns2rows_bit_reversed;

    let trace = fibonacci_rap_trace::<F>([FpE::one(), FpE::one()], trace_length);
    let air = FibonacciRAP::new(trace.num_rows(), pub_inputs, proof_options);
    let state = StarkMetalState::new().unwrap();
    let keccak_state = GpuKeccakMerkleState::new().unwrap();

    // Extract columns
    let t = Instant::now();
    let main_columns = trace.columns_main();
    println!(
        "  columns_main():    {:>10.2?}  ({} cols × {} rows)",
        t.elapsed(),
        main_columns.len(),
        main_columns[0].len()
    );

    // GPU interpolation
    let t = Instant::now();
    let main_trace_polys: Vec<Polynomial<FpE>> = main_columns
        .iter()
        .map(|col| {
            let coeffs = gpu_interpolate_fft::<F>(col, state.inner()).unwrap();
            Polynomial::new(&coeffs)
        })
        .collect();
    println!(
        "  GPU interpolate:   {:>10.2?}  ({} polys)",
        t.elapsed(),
        main_trace_polys.len()
    );

    // GPU LDE evaluation
    let blowup_factor = air.blowup_factor() as usize;
    let coset_offset = air.coset_offset();
    let t = Instant::now();
    let main_lde_evaluations: Vec<Vec<FpE>> = main_trace_polys
        .iter()
        .map(|poly| {
            gpu_evaluate_offset_fft::<F>(
                poly.coefficients(),
                blowup_factor,
                &coset_offset,
                state.inner(),
            )
            .unwrap()
        })
        .collect();
    let lde_rows = main_lde_evaluations[0].len();
    println!(
        "  GPU LDE eval:      {:>10.2?}  ({} cols × {} rows, blowup={})",
        t.elapsed(),
        main_lde_evaluations.len(),
        lde_rows,
        blowup_factor
    );

    // GPU Merkle commit breakdown
    let t = Instant::now();
    let rows = gpu_transpose_bitrev(&main_lde_evaluations, &keccak_state).unwrap();
    println!(
        "  GPU transpose+br:  {:>10.2?}  ({} rows × {} cols)",
        t.elapsed(),
        rows.len(),
        rows[0].len()
    );

    let t = Instant::now();
    let leaf_hashes = gpu_hash_leaves_goldilocks(&rows, &keccak_state).unwrap();
    println!(
        "  GPU leaf hash:     {:>10.2?}  ({} leaves)",
        t.elapsed(),
        leaf_hashes.len()
    );

    let t = Instant::now();
    let (nodes, _root) = gpu_build_merkle_tree(&leaf_hashes, &keccak_state).unwrap();
    println!(
        "  GPU tree build:    {:>10.2?}  ({} nodes)",
        t.elapsed(),
        nodes.len()
    );

    // Also time the CPU path for comparison
    let t = Instant::now();
    let _cpu_rows = columns2rows_bit_reversed(&main_lde_evaluations);
    println!(
        "  CPU transpose+br:  {:>10.2?}  (for comparison)",
        t.elapsed()
    );

    // CPU batch commit for comparison
    use lambdaworks_stark_gpu::metal::merkle::cpu_batch_commit;
    let cpu_rows = columns2rows_bit_reversed(&main_lde_evaluations);
    let t = Instant::now();
    let _ = cpu_batch_commit(&cpu_rows).unwrap();
    println!(
        "  CPU batch commit:  {:>10.2?}  (for comparison)",
        t.elapsed()
    );
}

fn profile_phase2(
    trace_length: usize,
    pub_inputs: &FibonacciRAPPublicInputs<F>,
    proof_options: &ProofOptions,
) {
    use lambdaworks_math::polynomial::Polynomial;
    use lambdaworks_stark_gpu::metal::constraint_eval::{
        gpu_evaluate_fibonacci_rap_constraints, FibRapConstraintState,
    };
    use lambdaworks_stark_gpu::metal::fft::gpu_evaluate_offset_fft;
    use lambdaworks_stark_gpu::metal::merkle::GpuKeccakMerkleState;
    use lambdaworks_stark_gpu::metal::phases::rap::gpu_round_1_goldilocks;
    use lambdaworks_stark_gpu::metal::state::StarkMetalState;
    use stark_platinum_prover::domain::Domain;
    use stark_platinum_prover::trace::LDETraceTable;

    let mut trace = fibonacci_rap_trace::<F>([FpE::one(), FpE::one()], trace_length);
    let air = FibonacciRAP::new(trace.num_rows(), pub_inputs, proof_options);
    let state = StarkMetalState::new().unwrap();
    let constraint_state = FibRapConstraintState::new().unwrap();
    let keccak_state = GpuKeccakMerkleState::new().unwrap();
    let domain = Domain::new(&air);

    // Run round 1 to get the input data
    let mut transcript = DefaultTranscript::<F>::new(&[]);
    let round_1 = gpu_round_1_goldilocks(
        &air,
        &mut trace,
        &domain,
        &mut transcript,
        &state,
        &keccak_state,
    )
    .unwrap();

    // Step 1: Sample beta and compute coefficients
    let t = Instant::now();
    let beta: FpE = transcript.sample_field_element();
    let num_boundary = air
        .boundary_constraints(&round_1.rap_challenges)
        .constraints
        .len();
    let num_transition = air.context().num_transition_constraints;
    let mut coefficients: Vec<FpE> = core::iter::successors(Some(FpE::one()), |x| Some(x * beta))
        .take(num_boundary + num_transition)
        .collect();
    let transition_coefficients: Vec<FpE> = coefficients.drain(..num_transition).collect();
    let boundary_coefficients = coefficients;
    println!("  Coefficients:      {:>10.2?}", t.elapsed());

    // Step 2: Build LDE trace
    let blowup_factor = air.blowup_factor() as usize;
    let t = Instant::now();
    let lde_trace = LDETraceTable::from_columns(
        round_1.main_lde_evaluations.clone(),
        round_1.aux_lde_evaluations.clone(),
        air.step_size(),
        blowup_factor,
    );
    println!("  LDE trace build:   {:>10.2?}", t.elapsed());

    // Step 3a: Boundary evaluations on CPU
    let num_lde_rows = lde_trace.num_rows();
    let t = Instant::now();
    let boundary_constraints = air.boundary_constraints(&round_1.rap_challenges);
    let mut zerofier_cache: std::collections::HashMap<usize, Vec<FpE>> =
        std::collections::HashMap::new();
    for bc in &boundary_constraints.constraints {
        zerofier_cache.entry(bc.step).or_insert_with(|| {
            let point = domain.trace_primitive_root.pow(bc.step as u64);
            let mut evals: Vec<FpE> = domain
                .lde_roots_of_unity_coset
                .iter()
                .map(|v| v - point)
                .collect();
            FpE::inplace_batch_inverse(&mut evals).unwrap();
            evals
        });
    }
    let boundary_zerofiers_refs: Vec<&Vec<FpE>> = boundary_constraints
        .constraints
        .iter()
        .map(|bc| zerofier_cache.get(&bc.step).unwrap())
        .collect();
    let boundary_poly_evals: Vec<Vec<FpE>> = boundary_constraints
        .constraints
        .iter()
        .map(|constraint| {
            if constraint.is_aux {
                (0..num_lde_rows)
                    .map(|row| lde_trace.get_aux(row, constraint.col) - constraint.value)
                    .collect()
            } else {
                (0..num_lde_rows)
                    .map(|row| lde_trace.get_main(row, constraint.col) - constraint.value)
                    .collect()
            }
        })
        .collect();
    let boundary_evals: Vec<FpE> = (0..num_lde_rows)
        .map(|i| {
            boundary_zerofiers_refs
                .iter()
                .zip(boundary_poly_evals.iter())
                .zip(boundary_coefficients.iter())
                .fold(FpE::zero(), |acc, ((z, bp), coeff)| {
                    acc + z[i] * coeff * bp[i]
                })
        })
        .collect();
    println!("  Boundary eval:     {:>10.2?}", t.elapsed());

    // Step 3b: Extract LDE columns
    let t = Instant::now();
    let main_col_0: Vec<FpE> = (0..num_lde_rows)
        .map(|r| *lde_trace.get_main(r, 0))
        .collect();
    let main_col_1: Vec<FpE> = (0..num_lde_rows)
        .map(|r| *lde_trace.get_main(r, 1))
        .collect();
    let aux_col_0: Vec<FpE> = (0..num_lde_rows)
        .map(|r| *lde_trace.get_aux(r, 0))
        .collect();
    println!("  Extract columns:   {:>10.2?}", t.elapsed());

    // Step 3c: GPU constraint evaluation
    let zerofier_evals = air.transition_zerofier_evaluations(&domain);
    let lde_step_size = air.step_size() * blowup_factor;
    let t = Instant::now();
    let constraint_evaluations = gpu_evaluate_fibonacci_rap_constraints(
        &main_col_0,
        &main_col_1,
        &aux_col_0,
        &zerofier_evals,
        &boundary_evals,
        &round_1.rap_challenges[0],
        &transition_coefficients,
        lde_step_size,
        Some(&constraint_state),
    )
    .unwrap();
    println!("  GPU constraints:   {:>10.2?}", t.elapsed());

    // Step 4: CPU IFFT (interpolate)
    let coset_offset = air.coset_offset();
    let t = Instant::now();
    let composition_poly =
        Polynomial::interpolate_offset_fft(&constraint_evaluations, &coset_offset).unwrap();
    println!("  CPU IFFT:          {:>10.2?}", t.elapsed());

    // Step 5: Break into parts
    let number_of_parts = air.composition_poly_degree_bound() / air.trace_length();
    let t = Instant::now();
    let composition_poly_parts = composition_poly.break_in_parts(number_of_parts);
    println!(
        "  Break parts:       {:>10.2?}  ({} parts)",
        t.elapsed(),
        composition_poly_parts.len()
    );

    // Step 6: GPU LDE eval
    let t = Instant::now();
    let lde_evaluations: Vec<Vec<FpE>> = composition_poly_parts
        .iter()
        .map(|part| {
            gpu_evaluate_offset_fft::<F>(
                part.coefficients(),
                blowup_factor,
                &coset_offset,
                state.inner(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    println!("  GPU LDE eval:      {:>10.2?}", t.elapsed());

    // Step 7: GPU Merkle commit (paired) - broken down
    {
        use lambdaworks_stark_gpu::metal::merkle::gpu_batch_commit_paired_goldilocks;

        // Time the combined function (flat_data + hash + tree in single command buffer)
        let t = Instant::now();
        let _ = gpu_batch_commit_paired_goldilocks(&lde_evaluations, &keccak_state).unwrap();
        let num_cols = lde_evaluations.len();
        let lde_len = lde_evaluations[0].len();
        println!(
            "  GPU paired commit: {:>10.2?}  ({} leaves × {} cols)",
            t.elapsed(),
            lde_len / 2,
            2 * num_cols
        );
    }
}

fn profile_phase4(
    trace_length: usize,
    pub_inputs: &FibonacciRAPPublicInputs<F>,
    proof_options: &ProofOptions,
) {
    use lambdaworks_stark_gpu::metal::constraint_eval::FibRapConstraintState;
    use lambdaworks_stark_gpu::metal::deep_composition::{
        gpu_compute_deep_composition_poly, DeepCompositionState,
    };
    use lambdaworks_stark_gpu::metal::fft::CosetShiftState;
    use lambdaworks_stark_gpu::metal::merkle::GpuKeccakMerkleState;
    use lambdaworks_stark_gpu::metal::phases::composition::gpu_round_2_goldilocks_merkle;
    use lambdaworks_stark_gpu::metal::phases::fri::FriFoldState;
    use lambdaworks_stark_gpu::metal::phases::ood::gpu_round_3;
    use lambdaworks_stark_gpu::metal::phases::rap::gpu_round_1_goldilocks;
    use lambdaworks_stark_gpu::metal::state::StarkMetalState;
    use stark_platinum_prover::domain::Domain;
    use stark_platinum_prover::traits::AIR;

    let mut trace = fibonacci_rap_trace::<F>([FpE::one(), FpE::one()], trace_length);
    let air = FibonacciRAP::new(trace.num_rows(), pub_inputs, proof_options);
    let state = StarkMetalState::new().unwrap();
    let constraint_state = FibRapConstraintState::new().unwrap();
    let deep_comp_state = DeepCompositionState::new().unwrap();
    let keccak_state = GpuKeccakMerkleState::new().unwrap();
    let coset_state = CosetShiftState::new().unwrap();
    let fri_fold_state = FriFoldState::new().unwrap();
    let domain = Domain::new(&air);
    let mut transcript = DefaultTranscript::<F>::new(&[]);

    let round_1 = gpu_round_1_goldilocks(
        &air,
        &mut trace,
        &domain,
        &mut transcript,
        &state,
        &keccak_state,
    )
    .unwrap();
    let round_2 = gpu_round_2_goldilocks_merkle(
        &air,
        &domain,
        &round_1,
        &mut transcript,
        &state,
        Some(&constraint_state),
        &keccak_state,
        &coset_state,
    )
    .unwrap();
    let round_3 = gpu_round_3(&air, &domain, &round_1, &round_2, &mut transcript).unwrap();

    // Step 1: Coefficients
    let t = Instant::now();
    let gamma = transcript.sample_field_element();
    let n_terms_comp = round_2.lde_composition_poly_evaluations.len();
    let num_terms_trace =
        air.context().transition_offsets.len() * air.step_size() * air.context().trace_columns;
    let mut deep_coeffs: Vec<FpE> = core::iter::successors(Some(FpE::one()), |x| Some(x * gamma))
        .take(n_terms_comp + num_terms_trace)
        .collect();
    let trace_term_coeffs: Vec<Vec<FpE>> = deep_coeffs
        .drain(..num_terms_trace)
        .collect::<Vec<_>>()
        .chunks(air.context().transition_offsets.len() * air.step_size())
        .map(|c| c.to_vec())
        .collect();
    let composition_gammas = deep_coeffs;
    println!("  Coefficients:      {:>10.2?}", t.elapsed());

    // Step 2: GPU DEEP composition
    let t = Instant::now();
    let deep_poly = gpu_compute_deep_composition_poly(
        &round_1,
        &round_2,
        &round_3,
        &domain,
        &composition_gammas,
        &trace_term_coeffs,
        &state,
        Some(&deep_comp_state),
    )
    .unwrap();
    println!("  GPU DEEP comp:     {:>10.2?}", t.elapsed());

    // Step 3: FRI commit phase (GPU FFT + GPU Keccak256 Merkle)
    use lambdaworks_stark_gpu::metal::phases::fri::gpu_fri_commit_phase_goldilocks;
    let domain_size = domain.lde_roots_of_unity_coset.len();
    let t = Instant::now();
    let (fri_last_value, fri_layers) = gpu_fri_commit_phase_goldilocks(
        domain.root_order as usize,
        deep_poly,
        &mut transcript,
        &domain.coset_offset,
        domain_size,
        &state,
        &keccak_state,
        &coset_state,
        &fri_fold_state,
    )
    .unwrap();
    println!(
        "  FRI commit (GPU):  {:>10.2?}  ({} layers)",
        t.elapsed(),
        fri_layers.len()
    );

    // Step 4: Grinding
    let security_bits = air.context().proof_options.grinding_factor;
    let t = Instant::now();
    let mut nonce = None;
    if security_bits > 0 {
        let nonce_value =
            stark_platinum_prover::grinding::generate_nonce(&transcript.state(), security_bits)
                .unwrap();
        transcript.append_bytes(&nonce_value.to_be_bytes());
        nonce = Some(nonce_value);
    }
    println!(
        "  Grinding:          {:>10.2?}  (bits={})",
        t.elapsed(),
        security_bits
    );

    // Step 5: Query indexes
    let number_of_queries = air.options().fri_number_of_queries;
    let t = Instant::now();
    let iotas: Vec<usize> = (0..number_of_queries)
        .map(|_| transcript.sample_u64((domain_size as u64) >> 1) as usize)
        .collect();
    println!(
        "  Query indexes:     {:>10.2?}  ({} queries)",
        t.elapsed(),
        iotas.len()
    );

    // Step 6: FRI query phase
    let t = Instant::now();
    let _query_list = stark_platinum_prover::fri::query_phase(&fri_layers, &iotas).unwrap();
    println!("  FRI query:         {:>10.2?}", t.elapsed());

    // Step 7: Open deep composition poly
    let t = Instant::now();
    // Can't call private function directly, but we can approximate
    let _ = &fri_last_value;
    let _ = &nonce;
    println!("  Proof openings:    (included in FRI query time)");
    let _ = t.elapsed();
}
