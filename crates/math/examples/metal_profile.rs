//! Metal GPU kernel profiler for lambdaworks.
//!
//! Profiles individual Metal kernel executions, measures throughput and memory
//! bandwidth utilization, identifies GPU/CPU crossover points, and prints a
//! detailed report with improvement suggestions.
//!
//! Run with: cargo run -p lambdaworks-math --features metal --example metal_profile --release

#![cfg(feature = "metal")]
#![allow(dead_code)]

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
use lambdaworks_math::unsigned_integer::element::UnsignedInteger;
use rand::{random, rngs::StdRng, Rng, SeedableRng};

type StarkF = Stark252PrimeField;
type StarkFE = FieldElement<StarkF>;
type GoldilocksF = Goldilocks64Field;
type GoldilocksFE = FieldElement<GoldilocksF>;
type M31FE = FieldElement<Mersenne31Field>;

// ============================================
// TIMING HELPERS
// ============================================

struct TimingResult {
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

    fn gpu_wins(&self) -> bool {
        self.gpu_ms < self.cpu_ms
    }

    fn gpu_throughput_melem_s(&self) -> f64 {
        self.n_elements as f64 / (self.gpu_ms / 1000.0) / 1_000_000.0
    }

    fn cpu_throughput_melem_s(&self) -> f64 {
        self.n_elements as f64 / (self.cpu_ms / 1000.0) / 1_000_000.0
    }

    fn effective_bandwidth_gb_s(&self) -> f64 {
        // FFT reads + writes each element ~log2(n) times across stages,
        // but the effective data movement is at least 2 * n * element_size
        // (one read + one write per element per pass).
        let data_bytes = 2.0 * self.n_elements as f64 * self.element_size_bytes as f64;
        let passes = self.order as f64; // log2(n) butterfly passes
        data_bytes * passes / (self.gpu_ms / 1000.0) / 1e9
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

const WARMUP_ITERS: usize = 3;
const BENCH_ITERS: usize = 10;

// ============================================
// PROFILING FUNCTIONS
// ============================================

fn profile_stark252_fft(state: &MetalState) -> Vec<TimingResult> {
    let orders: Vec<u32> = (10..=22).step_by(2).collect();
    let mut results = Vec::new();

    for &order in &orders {
        let n = 1usize << order;
        let input: Vec<StarkFE> = (0..n)
            .map(|_| StarkFE::new(UnsignedInteger { limbs: random() }))
            .collect();
        let twiddles = get_fft_twiddles::<StarkF>(order as u64, RootsConfig::BitReverse).unwrap();

        // Warmup
        for _ in 0..WARMUP_ITERS {
            let _ = lambdaworks_math::fft::cpu::ops::fft(&input, &twiddles);
            let _ = lambdaworks_math::fft::gpu::metal::ops::fft(&input, &twiddles, state);
        }

        // Bench CPU
        let mut cpu_times = Vec::with_capacity(BENCH_ITERS);
        for _ in 0..BENCH_ITERS {
            let start = Instant::now();
            let _ = lambdaworks_math::fft::cpu::ops::fft(&input, &twiddles);
            cpu_times.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        // Bench GPU
        let mut gpu_times = Vec::with_capacity(BENCH_ITERS);
        for _ in 0..BENCH_ITERS {
            let start = Instant::now();
            let _ = lambdaworks_math::fft::gpu::metal::ops::fft(&input, &twiddles, state);
            gpu_times.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        results.push(TimingResult {
            label: "Stark252 FFT".to_string(),
            order,
            n_elements: n,
            cpu_ms: median(&mut cpu_times),
            gpu_ms: median(&mut gpu_times),
            element_size_bytes: 32, // 256-bit field
        });
    }

    results
}

fn profile_goldilocks_fft(state: &MetalState) -> Vec<TimingResult> {
    let orders: Vec<u32> = (10..=22).step_by(2).collect();
    let mut results = Vec::new();

    for &order in &orders {
        let n = 1usize << order;
        let input: Vec<GoldilocksFE> = (0..n)
            .map(|_| GoldilocksFE::from(random::<u64>()))
            .collect();
        let twiddles =
            get_fft_twiddles::<GoldilocksF>(order as u64, RootsConfig::BitReverse).unwrap();

        // Warmup
        for _ in 0..WARMUP_ITERS {
            let _ = lambdaworks_math::fft::cpu::ops::fft(&input, &twiddles);
            let _ = lambdaworks_math::fft::gpu::metal::ops::fft(&input, &twiddles, state);
        }

        // Bench CPU
        let mut cpu_times = Vec::with_capacity(BENCH_ITERS);
        for _ in 0..BENCH_ITERS {
            let start = Instant::now();
            let _ = lambdaworks_math::fft::cpu::ops::fft(&input, &twiddles);
            cpu_times.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        // Bench GPU
        let mut gpu_times = Vec::with_capacity(BENCH_ITERS);
        for _ in 0..BENCH_ITERS {
            let start = Instant::now();
            let _ = lambdaworks_math::fft::gpu::metal::ops::fft(&input, &twiddles, state);
            gpu_times.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        results.push(TimingResult {
            label: "Goldilocks FFT".to_string(),
            order,
            n_elements: n,
            cpu_ms: median(&mut cpu_times),
            gpu_ms: median(&mut gpu_times),
            element_size_bytes: 8, // 64-bit field
        });
    }

    results
}

fn profile_mersenne31_cfft(state: &MetalState) -> Vec<TimingResult> {
    let orders: Vec<u32> = (10..=22).step_by(2).collect();
    let mut results = Vec::new();

    for &order in &orders {
        let n = 1usize << order;
        let mut rng = StdRng::seed_from_u64(0xCAFE + order as u64);
        let input: Vec<M31FE> = (0..n).map(|_| M31FE::from(&rng.gen::<u32>())).collect();

        // Warmup
        for _ in 0..WARMUP_ITERS {
            let _ = evaluate_cfft(input.clone());
            let _ =
                lambdaworks_math::circle::gpu::metal::ops::evaluate_cfft_gpu(input.clone(), state);
        }

        // Bench CPU
        let mut cpu_times = Vec::with_capacity(BENCH_ITERS);
        for _ in 0..BENCH_ITERS {
            let start = Instant::now();
            let _ = evaluate_cfft(input.clone());
            cpu_times.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        // Bench GPU
        let mut gpu_times = Vec::with_capacity(BENCH_ITERS);
        for _ in 0..BENCH_ITERS {
            let start = Instant::now();
            let _ =
                lambdaworks_math::circle::gpu::metal::ops::evaluate_cfft_gpu(input.clone(), state);
            gpu_times.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        results.push(TimingResult {
            label: "Mersenne31 CFFT (evaluate)".to_string(),
            order,
            n_elements: n,
            cpu_ms: median(&mut cpu_times),
            gpu_ms: median(&mut gpu_times),
            element_size_bytes: 4, // 32-bit field
        });
    }

    results
}

fn profile_mersenne31_icfft(state: &MetalState) -> Vec<TimingResult> {
    let orders: Vec<u32> = (10..=22).step_by(2).collect();
    let mut results = Vec::new();

    for &order in &orders {
        let n = 1usize << order;
        let mut rng = StdRng::seed_from_u64(0xCAFE + order as u64);
        let coeffs: Vec<M31FE> = (0..n).map(|_| M31FE::from(&rng.gen::<u32>())).collect();
        let evals = evaluate_cfft(coeffs);

        // Warmup
        for _ in 0..WARMUP_ITERS {
            let _ = interpolate_cfft(evals.clone());
            let _ = lambdaworks_math::circle::gpu::metal::ops::interpolate_cfft_gpu(
                evals.clone(),
                state,
            );
        }

        // Bench CPU
        let mut cpu_times = Vec::with_capacity(BENCH_ITERS);
        for _ in 0..BENCH_ITERS {
            let start = Instant::now();
            let _ = interpolate_cfft(evals.clone());
            cpu_times.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        // Bench GPU
        let mut gpu_times = Vec::with_capacity(BENCH_ITERS);
        for _ in 0..BENCH_ITERS {
            let start = Instant::now();
            let _ = lambdaworks_math::circle::gpu::metal::ops::interpolate_cfft_gpu(
                evals.clone(),
                state,
            );
            gpu_times.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        results.push(TimingResult {
            label: "Mersenne31 ICFFT (interpolate)".to_string(),
            order,
            n_elements: n,
            cpu_ms: median(&mut cpu_times),
            gpu_ms: median(&mut gpu_times),
            element_size_bytes: 4,
        });
    }

    results
}

fn profile_raw_cfft_butterflies(state: &MetalState) -> Vec<TimingResult> {
    let orders: Vec<u32> = (10..=22).step_by(2).collect();
    let mut results = Vec::new();

    for &order in &orders {
        let n = 1usize << order;
        let mut rng = StdRng::seed_from_u64(0xCAFE + order as u64);
        let input: Vec<M31FE> = (0..n).map(|_| M31FE::from(&rng.gen::<u32>())).collect();
        let coset = Coset::new_standard(order);
        let twiddles = get_twiddles(coset, TwiddlesConfig::Evaluation);

        // Warmup
        for _ in 0..WARMUP_ITERS {
            let mut cpu_data = input.clone();
            lambdaworks_math::circle::cfft::cfft(&mut cpu_data, &twiddles);
            let _ = lambdaworks_math::circle::gpu::metal::ops::cfft_gpu(&input, &twiddles, state);
        }

        // Bench CPU
        let mut cpu_times = Vec::with_capacity(BENCH_ITERS);
        for _ in 0..BENCH_ITERS {
            let mut data = input.clone();
            let start = Instant::now();
            lambdaworks_math::circle::cfft::cfft(&mut data, &twiddles);
            cpu_times.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        // Bench GPU
        let mut gpu_times = Vec::with_capacity(BENCH_ITERS);
        for _ in 0..BENCH_ITERS {
            let start = Instant::now();
            let _ = lambdaworks_math::circle::gpu::metal::ops::cfft_gpu(&input, &twiddles, state);
            gpu_times.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        results.push(TimingResult {
            label: "Mersenne31 CFFT butterflies (raw)".to_string(),
            order,
            n_elements: n,
            cpu_ms: median(&mut cpu_times),
            gpu_ms: median(&mut gpu_times),
            element_size_bytes: 4,
        });
    }

    results
}

// ============================================
// REPORT GENERATION
// ============================================

fn print_section(title: &str, results: &[TimingResult]) {
    println!("\n{}", "=".repeat(100));
    println!("  {title}");
    println!("{}", "=".repeat(100));
    println!(
        "{:<8} {:>12} {:>12} {:>12} {:>10} {:>12} {:>14}",
        "2^order", "N", "CPU (ms)", "GPU (ms)", "Speedup", "GPU Melem/s", "Bandwidth GB/s"
    );
    println!("{}", "-".repeat(100));

    let mut crossover = None;
    let mut prev_gpu_wins = false;

    for (i, r) in results.iter().enumerate() {
        let winner = if r.gpu_wins() { "<< GPU" } else { "" };
        println!(
            "{:<8} {:>12} {:>12.3} {:>12.3} {:>9.2}x {:>12.1} {:>14.2} {}",
            format!("2^{}", r.order),
            r.n_elements,
            r.cpu_ms,
            r.gpu_ms,
            r.speedup(),
            r.gpu_throughput_melem_s(),
            r.effective_bandwidth_gb_s(),
            winner,
        );

        let gpu_wins = r.gpu_wins();
        if i > 0 && gpu_wins != prev_gpu_wins {
            crossover = Some(r.order);
        }
        prev_gpu_wins = gpu_wins;
    }

    if let Some(order) = crossover {
        println!(
            "\n  >> GPU/CPU crossover near 2^{order} ({} elements)",
            1u64 << order
        );
    } else if results.last().map_or(false, |r| r.gpu_wins()) {
        println!("\n  >> GPU wins across all tested sizes");
    } else {
        println!("\n  >> CPU wins across all tested sizes (GPU overhead dominates)");
    }
}

fn print_improvement_suggestions(all_results: &[Vec<TimingResult>]) {
    println!("\n{}", "=".repeat(100));
    println!("  IMPROVEMENT SUGGESTIONS");
    println!("{}", "=".repeat(100));

    // Check for high GPU overhead at small sizes
    let mut has_small_size_overhead = false;
    for results in all_results {
        if let Some(r) = results.first() {
            if !r.gpu_wins() && r.gpu_ms > r.cpu_ms * 2.0 {
                has_small_size_overhead = true;
            }
        }
    }

    if has_small_size_overhead {
        println!("\n1. HIGH GPU LAUNCH OVERHEAD AT SMALL SIZES");
        println!("   The Metal GPU has significant fixed overhead per command buffer.");
        println!("   Suggestions:");
        println!("   - Add a GPU crossover threshold: fall back to CPU for small inputs");
        println!("   - Batch multiple small FFTs into a single command buffer");
        println!("   - Reuse command encoders across butterfly stages (already done for CFFT)");
    }

    // Analyze bandwidth utilization
    let mut low_bandwidth = false;
    for results in all_results {
        for r in results {
            // Apple M1/M2 has ~200 GB/s memory bandwidth. If we're under 50 GB/s, there's room.
            if r.gpu_wins() && r.effective_bandwidth_gb_s() < 50.0 {
                low_bandwidth = true;
            }
        }
    }

    if low_bandwidth {
        println!("\n2. LOW MEMORY BANDWIDTH UTILIZATION");
        println!("   Apple Silicon has ~200 GB/s unified memory bandwidth.");
        println!("   Current kernels use well under 25% of available bandwidth.");
        println!("   Suggestions:");
        println!("   - Use threadgroup (shared) memory for twiddle factors within a stage");
        println!(
            "   - Coalesce memory accesses: process multiple butterfly stages per kernel launch"
        );
        println!("   - Use SIMD group operations for reductions within a warp/SIMD group");
    }

    // FFT architecture suggestions
    println!("\n3. KERNEL LAUNCH PER STAGE");
    println!("   Current FFT dispatches one kernel per butterfly stage (log2(n) dispatches).");
    println!("   Each dispatch has ~5-15us of overhead on Metal.");
    println!("   Suggestions:");
    println!("   - Fuse early stages: the first few stages fit in threadgroup memory");
    println!("   - For Mersenne31 (32-bit): 256 elements fit in 1KB of threadgroup memory");
    println!("   - Implement a multi-stage kernel that processes 4-8 stages internally");

    println!("\n4. BUFFER ALLOCATION");
    println!("   Current implementation allocates new Metal buffers per call.");
    println!("   Suggestions:");
    println!("   - Pool and reuse Metal buffers across calls (amortize allocation)");
    println!("   - Pre-allocate twiddle factor buffers for common sizes");
    println!("   - Use private storage mode + blit for data that doesn't need CPU access");

    println!("\n5. FIELD-SPECIFIC OPTIMIZATIONS");
    println!("   Mersenne31 (p = 2^31 - 1):");
    println!("   - Already uses fast Mersenne reduction (shift + mask)");
    println!("   - Consider: packed 2x Mersenne31 in a single 64-bit register");
    println!("   - Consider: SIMD group shuffle for butterfly communication");
    println!("   Stark252 (256-bit):");
    println!("   - 256-bit multiply is very expensive on GPU (multiple u64 multiplies)");
    println!("   - Consider: Montgomery multiplication with u32 limbs for better GPU occupancy");
    println!("   - Consider: using threadgroup memory for intermediate limb products");
}

fn print_device_info(state: &MetalState) {
    let device = &state.device;
    let max_tg = device.max_threads_per_threadgroup();
    println!("{}", "=".repeat(100));
    println!("  METAL GPU DEVICE INFO");
    println!("{}", "=".repeat(100));
    println!("  Device name:             {}", device.name());
    println!(
        "  Max threads/threadgroup: {}x{}x{}",
        max_tg.width, max_tg.height, max_tg.depth
    );
    println!(
        "  Max buffer length:       {} MB",
        device.max_buffer_length() / (1024 * 1024)
    );
    println!(
        "  Recommended working set: {} MB",
        device.recommended_max_working_set_size() / (1024 * 1024)
    );
    println!("  Unified memory:          {}", device.has_unified_memory());
}

fn main() {
    println!("\nLambdaworks Metal GPU Profiler");
    println!("==============================\n");

    let state = match MetalState::new(None) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to initialize Metal: {:?}", e);
            eprintln!("Make sure you're running on macOS with Metal support.");
            std::process::exit(1);
        }
    };

    print_device_info(&state);

    println!("\nRunning profiling (this may take a few minutes)...\n");

    let stark_results = profile_stark252_fft(&state);
    print_section("Stark252 FFT (256-bit field, GPU vs CPU)", &stark_results);

    let goldilocks_results = profile_goldilocks_fft(&state);
    print_section(
        "Goldilocks FFT (64-bit field, GPU vs CPU)",
        &goldilocks_results,
    );

    let cfft_results = profile_mersenne31_cfft(&state);
    print_section(
        "Mersenne31 CFFT evaluate (32-bit field, full pipeline, GPU vs CPU)",
        &cfft_results,
    );

    let icfft_results = profile_mersenne31_icfft(&state);
    print_section(
        "Mersenne31 ICFFT interpolate (32-bit field, full pipeline, GPU vs CPU)",
        &icfft_results,
    );

    let butterfly_results = profile_raw_cfft_butterflies(&state);
    print_section(
        "Mersenne31 CFFT raw butterflies (GPU kernel only, GPU vs CPU)",
        &butterfly_results,
    );

    print_improvement_suggestions(&[
        stark_results,
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
