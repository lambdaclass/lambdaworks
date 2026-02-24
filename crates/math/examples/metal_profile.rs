//! Metal GPU kernel profiler for lambdaworks.
//!
//! Profiles individual Metal kernel executions, measures throughput and memory
//! bandwidth utilization, identifies GPU/CPU crossover points, and prints a
//! detailed report with improvement suggestions.
//!
//! # Purpose
//!
//! This profiler is designed for **performance measurement and optimization only**.
//! It generates random data for benchmarking purposes and should NOT be used with
//! sensitive or production data. The profiling operations (including sorting and
//! statistical analysis) are not designed to be constant-time and may leak timing
//! information.
//!
//! For production cryptographic operations, use the library's standard APIs with
//! appropriate security considerations.
//!
//! # Usage
//!
//! Run with: cargo run -p lambdaworks-math --features metal --example metal_profile --release

#![allow(dead_code)]

#[cfg(feature = "metal")]
mod profiler {
    use std::time::Instant;

    use lambdaworks_gpu::metal::abstractions::state::MetalState;
    use lambdaworks_math::circle::cosets::Coset;
    use lambdaworks_math::circle::polynomial::{evaluate_cfft, interpolate_cfft};
    use lambdaworks_math::circle::twiddles::{get_twiddles, TwiddlesConfig};
    use lambdaworks_math::fft::cpu::roots_of_unity::get_twiddles as get_fft_twiddles;
    use lambdaworks_math::field::element::FieldElement;
    use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
    use lambdaworks_math::field::fields::mersenne31::field::Mersenne31Field;
    use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;
    use lambdaworks_math::field::traits::RootsConfig;
    use rand::random;

    type StarkF = Stark252PrimeField;
    type StarkFE = FieldElement<StarkF>;
    type GoldilocksF = Goldilocks64Field;
    type GoldilocksFE = FieldElement<GoldilocksF>;
    type M31FE = FieldElement<Mersenne31Field>;

    // ============================================
    // TIMING HELPERS
    // ============================================

    #[derive(Clone)]
    pub(crate) struct TimingResult {
        label: String,
        order: u32,
        n_elements: usize,
        cpu_ms: f64,
        gpu_ms: f64,
        element_size_bytes: usize,
    }

    impl TimingResult {
        fn speedup(&self) -> f64 {
            self.cpu_ms / self.gpu_ms
        }

        fn throughput_melems_per_sec(&self, is_gpu: bool) -> f64 {
            let time_ms = if is_gpu { self.gpu_ms } else { self.cpu_ms };
            (self.n_elements as f64 / 1_000_000.0) / (time_ms / 1000.0)
        }

        fn memory_bandwidth_gb_per_sec(&self, is_gpu: bool) -> f64 {
            let time_ms = if is_gpu { self.gpu_ms } else { self.cpu_ms };
            let total_bytes = self.n_elements * self.element_size_bytes;
            // Read + write = 2x memory accesses
            let total_gb = (total_bytes * 2) as f64 / 1_073_741_824.0;
            total_gb / (time_ms / 1000.0)
        }
    }

    fn time_gpu_stark_fft<F: Fn(&MetalState, &[StarkFE]) -> Vec<StarkFE>>(
        state: &MetalState,
        input: &[StarkFE],
        gpu_fn: F,
    ) -> f64 {
        // Warmup
        for _ in 0..2 {
            let _ = gpu_fn(state, input);
        }

        // Measure
        let n_runs = 5;
        let start = Instant::now();
        for _ in 0..n_runs {
            let _ = gpu_fn(state, input);
        }
        let elapsed = start.elapsed();
        elapsed.as_secs_f64() * 1000.0 / n_runs as f64
    }

    fn time_cpu_stark_fft<F: Fn(&[StarkFE]) -> Vec<StarkFE>>(input: &[StarkFE], cpu_fn: F) -> f64 {
        // Warmup
        for _ in 0..2 {
            let _ = cpu_fn(input);
        }

        // Measure
        let n_runs = 5;
        let start = Instant::now();
        for _ in 0..n_runs {
            let _ = cpu_fn(input);
        }
        let elapsed = start.elapsed();
        elapsed.as_secs_f64() * 1000.0 / n_runs as f64
    }

    fn time_gpu_goldilocks_fft<F: Fn(&MetalState, &[GoldilocksFE]) -> Vec<GoldilocksFE>>(
        state: &MetalState,
        input: &[GoldilocksFE],
        gpu_fn: F,
    ) -> f64 {
        // Warmup
        for _ in 0..2 {
            let _ = gpu_fn(state, input);
        }

        // Measure
        let n_runs = 5;
        let start = Instant::now();
        for _ in 0..n_runs {
            let _ = gpu_fn(state, input);
        }
        let elapsed = start.elapsed();
        elapsed.as_secs_f64() * 1000.0 / n_runs as f64
    }

    fn time_cpu_goldilocks_fft<F: Fn(&[GoldilocksFE]) -> Vec<GoldilocksFE>>(
        input: &[GoldilocksFE],
        cpu_fn: F,
    ) -> f64 {
        // Warmup
        for _ in 0..2 {
            let _ = cpu_fn(input);
        }

        // Measure
        let n_runs = 5;
        let start = Instant::now();
        for _ in 0..n_runs {
            let _ = cpu_fn(input);
        }
        let elapsed = start.elapsed();
        elapsed.as_secs_f64() * 1000.0 / n_runs as f64
    }

    fn time_gpu_m31_cfft<F: Fn(&MetalState, &[M31FE]) -> Vec<M31FE>>(
        state: &MetalState,
        input: &[M31FE],
        gpu_fn: F,
    ) -> f64 {
        // Warmup
        for _ in 0..2 {
            let _ = gpu_fn(state, input);
        }

        // Measure
        let n_runs = 5;
        let start = Instant::now();
        for _ in 0..n_runs {
            let _ = gpu_fn(state, input);
        }
        let elapsed = start.elapsed();
        elapsed.as_secs_f64() * 1000.0 / n_runs as f64
    }

    fn time_cpu_m31_cfft<F: Fn(&[M31FE]) -> Vec<M31FE>>(input: &[M31FE], cpu_fn: F) -> f64 {
        // Warmup
        for _ in 0..2 {
            let _ = cpu_fn(input);
        }

        // Measure
        let n_runs = 5;
        let start = Instant::now();
        for _ in 0..n_runs {
            let _ = cpu_fn(input);
        }
        let elapsed = start.elapsed();
        elapsed.as_secs_f64() * 1000.0 / n_runs as f64
    }

    // ============================================
    // FFT PROFILERS
    // ============================================

    fn profile_stark_fft(state: &MetalState, order: u32) -> TimingResult {
        let n = 1 << order;
        let input: Vec<StarkFE> = (0..n).map(|_| StarkFE::from(random::<u64>())).collect();

        let twiddles_cpu = get_fft_twiddles::<StarkF>(n, RootsConfig::Natural).unwrap();
        let cpu_fn = |input: &[StarkFE]| {
            use lambdaworks_math::fft::cpu::ops as fft_ops;
            let mut result = input.to_vec();
            result = fft_ops::fft(&result, &twiddles_cpu).unwrap();
            result
        };
        let cpu_ms = time_cpu_stark_fft(&input, cpu_fn);

        let gpu_fn = |state: &MetalState, input: &[StarkFE]| {
            use lambdaworks_math::fft::gpu::metal::ops as metal_ops;
            metal_ops::fft(input, &twiddles_cpu, state).unwrap()
        };
        let gpu_ms = time_gpu_stark_fft(state, &input, gpu_fn);

        TimingResult {
            label: "Stark252 FFT".to_string(),
            order,
            n_elements: n as usize,
            cpu_ms,
            gpu_ms,
            element_size_bytes: 32,
        }
    }

    fn profile_stark_ifft(state: &MetalState, order: u32) -> TimingResult {
        let n = 1 << order;
        let input: Vec<StarkFE> = (0..n).map(|_| StarkFE::from(random::<u64>())).collect();

        let twiddles_cpu = get_fft_twiddles::<StarkF>(n, RootsConfig::NaturalInversed).unwrap();
        let cpu_fn = |input: &[StarkFE]| {
            use lambdaworks_math::fft::cpu::ops as fft_ops;
            let mut result = input.to_vec();
            result = fft_ops::fft(&result, &twiddles_cpu).unwrap();
            result
        };
        let cpu_ms = time_cpu_stark_fft(&input, cpu_fn);

        let gpu_fn = |state: &MetalState, input: &[StarkFE]| {
            use lambdaworks_math::fft::gpu::metal::ops as metal_ops;
            metal_ops::fft(input, &twiddles_cpu, state).unwrap()
        };
        let gpu_ms = time_gpu_stark_fft(state, &input, gpu_fn);

        TimingResult {
            label: "Stark252 IFFT".to_string(),
            order,
            n_elements: n as usize,
            cpu_ms,
            gpu_ms,
            element_size_bytes: 32,
        }
    }

    fn profile_goldilocks_fft(state: &MetalState, order: u32) -> TimingResult {
        let n = 1 << order;
        let input: Vec<GoldilocksFE> = (0..n)
            .map(|_| GoldilocksFE::from(random::<u64>()))
            .collect();

        let twiddles_cpu = get_fft_twiddles::<GoldilocksF>(n, RootsConfig::Natural).unwrap();
        let cpu_fn = |input: &[GoldilocksFE]| {
            use lambdaworks_math::fft::cpu::ops as fft_ops;
            let mut result = input.to_vec();
            result = fft_ops::fft(&result, &twiddles_cpu).unwrap();
            result
        };
        let cpu_ms = time_cpu_goldilocks_fft(&input, cpu_fn);

        let gpu_fn = |state: &MetalState, input: &[GoldilocksFE]| {
            use lambdaworks_math::fft::gpu::metal::ops as metal_ops;
            metal_ops::fft(input, &twiddles_cpu, state).unwrap()
        };
        let gpu_ms = time_gpu_goldilocks_fft(state, &input, gpu_fn);

        TimingResult {
            label: "Goldilocks FFT".to_string(),
            order,
            n_elements: n as usize,
            cpu_ms,
            gpu_ms,
            element_size_bytes: 8,
        }
    }

    fn profile_goldilocks_ifft(state: &MetalState, order: u32) -> TimingResult {
        let n = 1 << order;
        let input: Vec<GoldilocksFE> = (0..n)
            .map(|_| GoldilocksFE::from(random::<u64>()))
            .collect();

        let twiddles_cpu =
            get_fft_twiddles::<GoldilocksF>(n, RootsConfig::NaturalInversed).unwrap();
        let cpu_fn = |input: &[GoldilocksFE]| {
            use lambdaworks_math::fft::cpu::ops as fft_ops;
            let mut result = input.to_vec();
            result = fft_ops::fft(&result, &twiddles_cpu).unwrap();
            result
        };
        let cpu_ms = time_cpu_goldilocks_fft(&input, cpu_fn);

        let gpu_fn = |state: &MetalState, input: &[GoldilocksFE]| {
            use lambdaworks_math::fft::gpu::metal::ops as metal_ops;
            metal_ops::fft(input, &twiddles_cpu, state).unwrap()
        };
        let gpu_ms = time_gpu_goldilocks_fft(state, &input, gpu_fn);

        TimingResult {
            label: "Goldilocks IFFT".to_string(),
            order,
            n_elements: n as usize,
            cpu_ms,
            gpu_ms,
            element_size_bytes: 8,
        }
    }

    // ============================================
    // CFFT PROFILERS
    // ============================================

    fn profile_m31_cfft(state: &MetalState, order: u32) -> TimingResult {
        let coset_size = 1 << order;
        let _coset = Coset::<Mersenne31Field>::new_standard(order);
        let input: Vec<M31FE> = (0..coset_size)
            .map(|_| M31FE::from(&random::<u32>()))
            .collect();

        let cpu_fn = |input: &[M31FE]| {
            let result = evaluate_cfft(input.to_vec());
            result
        };
        let cpu_ms = time_cpu_m31_cfft(&input, cpu_fn);

        let gpu_fn = |state: &MetalState, input: &[M31FE]| {
            use lambdaworks_math::circle::gpu::metal::ops as metal_ops;
            metal_ops::evaluate_cfft_gpu(input.to_vec(), state).unwrap()
        };
        let gpu_ms = time_gpu_m31_cfft(state, &input, gpu_fn);

        TimingResult {
            label: "Mersenne31 CFFT".to_string(),
            order,
            n_elements: coset_size,
            cpu_ms,
            gpu_ms,
            element_size_bytes: 4,
        }
    }

    fn profile_m31_icfft(state: &MetalState, order: u32) -> TimingResult {
        let coset_size = 1 << order;
        let _coset = Coset::<Mersenne31Field>::new_standard(order);
        let input: Vec<M31FE> = (0..coset_size)
            .map(|_| M31FE::from(&random::<u32>()))
            .collect();

        let cpu_fn = |input: &[M31FE]| {
            let result = interpolate_cfft(input.to_vec());
            result
        };
        let cpu_ms = time_cpu_m31_cfft(&input, cpu_fn);

        let gpu_fn = |state: &MetalState, input: &[M31FE]| {
            use lambdaworks_math::circle::gpu::metal::ops as metal_ops;
            metal_ops::interpolate_cfft_gpu(input.to_vec(), state).unwrap()
        };
        let gpu_ms = time_gpu_m31_cfft(state, &input, gpu_fn);

        TimingResult {
            label: "Mersenne31 ICFFT".to_string(),
            order,
            n_elements: coset_size,
            cpu_ms,
            gpu_ms,
            element_size_bytes: 4,
        }
    }

    // ============================================
    // BUTTERFLY PROFILER (M31)
    // ============================================

    fn profile_m31_butterfly(state: &MetalState, order: u32) -> TimingResult {
        let n = 1 << order;
        let input: Vec<M31FE> = (0..n).map(|_| M31FE::from(&random::<u32>())).collect();
        let coset = Coset::new_standard(order);
        let twiddles = get_twiddles(coset, TwiddlesConfig::Evaluation);

        // CPU: CFFT butterflies
        let cpu_fn = |input: &[M31FE]| {
            let mut result = input.to_vec();
            lambdaworks_math::circle::cfft::cfft(&mut result, &twiddles);
            result
        };
        let cpu_ms = time_cpu_m31_cfft(&input, cpu_fn);

        // GPU: CFFT butterflies (raw GPU kernel without full pipeline)
        let gpu_fn = |state: &MetalState, input: &[M31FE]| {
            use lambdaworks_math::circle::gpu::metal::ops as metal_ops;
            metal_ops::cfft_gpu(input, &twiddles, state).unwrap()
        };
        let gpu_ms = time_gpu_m31_cfft(state, &input, gpu_fn);

        TimingResult {
            label: "Mersenne31 CFFT Butterflies".to_string(),
            order,
            n_elements: n,
            cpu_ms,
            gpu_ms,
            element_size_bytes: 4,
        }
    }

    // ============================================
    // REPORTING
    // ============================================

    fn print_results_table(results: &[Vec<TimingResult>]) {
        println!("\n{}", "=".repeat(150));
        println!(
            "{:<25} {:>8} {:>12} {:>12} {:>12} {:>15} {:>15} {:>15} {:>15}",
            "Operation",
            "Order",
            "Elements",
            "CPU (ms)",
            "GPU (ms)",
            "Speedup",
            "CPU (Melem/s)",
            "GPU (Melem/s)",
            "GPU BW (GB/s)"
        );
        println!("{}", "=".repeat(150));

        for group in results {
            for result in group {
                println!(
                    "{:<25} {:>8} {:>12} {:>12.3} {:>12.3} {:>14.2}x {:>15.1} {:>15.1} {:>15.1}",
                    result.label,
                    result.order,
                    result.n_elements,
                    result.cpu_ms,
                    result.gpu_ms,
                    result.speedup(),
                    result.throughput_melems_per_sec(false),
                    result.throughput_melems_per_sec(true),
                    result.memory_bandwidth_gb_per_sec(true),
                );
            }
            println!("{}", "-".repeat(150));
        }
    }

    fn analyze_results(results: &[Vec<TimingResult>]) {
        println!("\n{}", "=".repeat(100));
        println!("  PERFORMANCE ANALYSIS");
        println!("{}", "=".repeat(100));

        for group in results {
            if group.is_empty() {
                continue;
            }

            let operation_name = &group[0].label;
            println!("\n{}", operation_name);
            println!("{}", "-".repeat(operation_name.len()));

            // Find crossover point (where GPU becomes faster)
            let mut crossover = None;
            for (i, result) in group.iter().enumerate() {
                if result.speedup() > 1.0 {
                    crossover = Some(i);
                    break;
                }
            }

            match crossover {
                Some(idx) => {
                    let result = &group[idx];
                    println!(
                        "✓ GPU advantage starts at 2^{} elements ({} elements)",
                        result.order, result.n_elements
                    );
                    println!(
                        "  At this size: {:.2}x speedup, {:.1} Melem/s throughput",
                        result.speedup(),
                        result.throughput_melems_per_sec(true)
                    );
                }
                None => {
                    println!("⚠ GPU never becomes faster than CPU in tested range");
                    if let Some(last) = group.last() {
                        if last.speedup() > 0.5 {
                            println!(
                                "  At 2^{}: {:.2}x speedup (close to crossover)",
                                last.order,
                                last.speedup()
                            );
                        }
                    }
                }
            }

            // Peak performance
            if let Some(best) = group.iter().max_by(|a, b| {
                a.throughput_melems_per_sec(true)
                    .partial_cmp(&b.throughput_melems_per_sec(true))
                    .unwrap()
            }) {
                println!(
                    "✓ Peak GPU throughput: {:.1} Melem/s at 2^{} elements",
                    best.throughput_melems_per_sec(true),
                    best.order
                );
                println!(
                    "  Memory bandwidth: {:.1} GB/s",
                    best.memory_bandwidth_gb_per_sec(true)
                );
            }
        }
    }

    fn print_device_info(state: &MetalState) {
        println!("\n{}", "=".repeat(100));
        println!("  METAL DEVICE INFO");
        println!("{}", "=".repeat(100));

        let device = &state.device;
        println!("  Device name:             {}", device.name());
        println!(
            "  Max threads per group:   {}",
            device.max_threads_per_threadgroup().width
        );
        println!(
            "  Max buffer length:       {} bytes",
            device.max_buffer_length()
        );
        println!("  Unified memory:          {}", device.has_unified_memory());
    }

    pub fn run_profiler() {
        println!("\nLambdaworks Metal GPU Profiler");
        println!("==============================\n");

        let state = match MetalState::new(None) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Failed to initialize Metal state: {:?}", e);
                eprintln!("Make sure you're running on a system with Metal support (macOS/iOS).");
                std::process::exit(1);
            }
        };

        print_device_info(&state);

        // Input size variations for profiling:
        // - Includes power-of-2 sizes (12, 14, 16, ...) for standard benchmarking
        // - Adds intermediate sizes (13, 15, 17, ...) for finer-grained crossover analysis
        // - Smaller sizes (10, 11) help identify CPU/GPU threshold (typically around 2^14)
        // - Larger sizes (20, 21, 22) demonstrate GPU advantages at scale
        //
        // Order N corresponds to 2^N elements:
        //   10 → 1K,  11 → 2K,  12 → 4K,  13 → 8K,  14 → 16K (GPU threshold),
        //   15 → 32K, 16 → 64K, 17 → 128K, 18 → 256K, 19 → 512K,
        //   20 → 1M,  21 → 2M,  22 → 4M
        let orders = vec![10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22];

        println!("\n{}", "=".repeat(100));
        println!("  PROFILING STARK252 FIELD FFT/IFFT");
        println!("{}", "=".repeat(100));

        let mut stark_fft_results = Vec::new();
        for &order in &orders {
            println!("  Profiling Stark252 FFT with order {}...", order);
            stark_fft_results.push(profile_stark_fft(&state, order));
        }

        let mut stark_ifft_results = Vec::new();
        for &order in &orders {
            println!("  Profiling Stark252 IFFT with order {}...", order);
            stark_ifft_results.push(profile_stark_ifft(&state, order));
        }

        println!("\n{}", "=".repeat(100));
        println!("  PROFILING GOLDILOCKS FIELD FFT/IFFT");
        println!("{}", "=".repeat(100));

        let mut goldilocks_results = Vec::new();
        for &order in &orders {
            println!("  Profiling Goldilocks FFT with order {}...", order);
            goldilocks_results.push(profile_goldilocks_fft(&state, order));
        }

        for &order in &orders {
            println!("  Profiling Goldilocks IFFT with order {}...", order);
            goldilocks_results.push(profile_goldilocks_ifft(&state, order));
        }

        println!("\n{}", "=".repeat(100));
        println!("  PROFILING MERSENNE31 CIRCLE FFT");
        println!("{}", "=".repeat(100));

        let mut cfft_results = Vec::new();
        for &order in &orders {
            println!("  Profiling Mersenne31 CFFT with order {}...", order);
            cfft_results.push(profile_m31_cfft(&state, order));
        }

        let mut icfft_results = Vec::new();
        for &order in &orders {
            println!("  Profiling Mersenne31 ICFFT with order {}...", order);
            icfft_results.push(profile_m31_icfft(&state, order));
        }

        println!("\n{}", "=".repeat(100));
        println!("  PROFILING MERSENNE31 BUTTERFLY");
        println!("{}", "=".repeat(100));

        let mut butterfly_results = Vec::new();
        for &order in &orders {
            println!("  Profiling Mersenne31 Butterfly with order {}...", order);
            butterfly_results.push(profile_m31_butterfly(&state, order));
        }

        print_results_table(&vec![
            stark_fft_results.clone(),
            stark_ifft_results.clone(),
            goldilocks_results.clone(),
            cfft_results.clone(),
            icfft_results.clone(),
            butterfly_results.clone(),
        ]);

        analyze_results(&vec![
            stark_fft_results,
            stark_ifft_results,
            goldilocks_results,
            cfft_results,
            icfft_results,
            butterfly_results,
        ]);

        println!("\n{}", "=".repeat(100));
        println!("  PROFILING COMPLETE");
        println!("{}", "=".repeat(100));
        println!("\nTo run criterion benchmarks for statistical analysis:");
        println!("  cargo bench -p lambdaworks-math --features metal --bench metal_fft");
        println!("  cargo bench -p lambdaworks-math --features metal --bench metal_cfft");
        println!();
    }
}

#[cfg(feature = "metal")]
fn main() {
    profiler::run_profiler();
}

#[cfg(not(feature = "metal"))]
fn main() {
    eprintln!("This example requires the 'metal' feature to be enabled.");
    eprintln!("Run with: cargo run -p lambdaworks-math --features metal --example metal_profile --release");
    std::process::exit(1);
}
