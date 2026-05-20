//! Criterion benchmark for the Metal Bowers-G fused NTT + coset LDE.
//!
//! Two groups:
//!   * `metal_bowers_lde_goldilocks` — the new fused Bowers LDE at STARK shapes.
//!   * `bowers_vs_baseline_2pow20x64` — head-to-head against the existing
//!     radix-2 DIT FFT (`fft_buffer_to_buffer`) run once per column.
//!
//! Run with:
//!   cargo bench -p lambdaworks-math --features metal --bench criterion_metal_bowers_lde

#![cfg(all(target_os = "macos", feature = "metal"))]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use lambdaworks_gpu::metal::abstractions::state::MetalState;
use lambdaworks_math::fft::cpu::roots_of_unity::get_twiddles;
use lambdaworks_math::fft::gpu::metal::bowers::dispatcher::metal_bowers_lde;
use lambdaworks_math::fft::gpu::metal::bowers::twiddles::{
    cached_bowers_twiddles_goldilocks, cached_coset_powers_goldilocks,
};
use lambdaworks_math::fft::gpu::metal::ops::fft_buffer_to_buffer;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;
use lambdaworks_math::field::traits::RootsConfig;

type Fp = FieldElement<Goldilocks64Field>;

fn bit_reverse<T>(data: &mut [T]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    let log_n = n.trailing_zeros();
    for i in 0..n {
        let j = ((i as u32).reverse_bits() >> (u32::BITS - log_n)) as usize;
        if i < j {
            data.swap(i, j);
        }
    }
}

fn bench_bowers_lde(c: &mut Criterion) {
    let state = MetalState::new(None).expect("metal state");
    let coset = Fp::from(7u64);
    let mut group = c.benchmark_group("metal_bowers_lde_goldilocks");

    for &(log_n, num_cols) in &[(18u32, 64u32), (20u32, 64u32), (20u32, 256u32)] {
        let n = 1usize << log_n;
        let total = n * num_cols as usize;

        let input: Vec<u64> = (0..total as u64)
            .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .collect();
        let cols_buf = state.alloc_buffer_data(&input);
        let mut out_buf = state.alloc_buffer::<u64>(total);

        let twiddles = cached_bowers_twiddles_goldilocks(log_n);
        let tw_u64: Vec<u64> = twiddles.iter().map(|x| *x.value()).collect();
        let tw_buf = state.alloc_buffer_data(&tw_u64);

        let coset_powers = cached_coset_powers_goldilocks(log_n, coset);
        let mut coset_u64: Vec<u64> = coset_powers.iter().map(|x| *x.value()).collect();
        bit_reverse(&mut coset_u64);
        let coset_buf = state.alloc_buffer_data(&coset_u64);

        group.throughput(Throughput::Elements(total as u64));
        group.bench_with_input(
            BenchmarkId::new("lde", format!("log_n_{log_n}_cols_{num_cols}")),
            &(log_n, num_cols),
            |b, &(log_n, num_cols)| {
                b.iter(|| {
                    metal_bowers_lde(
                        &cols_buf,
                        &tw_buf,
                        &coset_buf,
                        &mut out_buf,
                        log_n,
                        num_cols,
                        &state,
                    )
                    .unwrap();
                });
            },
        );
    }
    group.finish();
}

/// Head-to-head at 2^20 x 64: the new fused Bowers LDE (one dispatch for all
/// columns) versus the existing radix-2 DIT FFT run once per column.
fn bench_bowers_vs_baseline(c: &mut Criterion) {
    let state = MetalState::new(None).expect("metal state");
    let log_n = 20u32;
    let n = 1usize << log_n;
    let num_cols = 64u32;
    let total = n * num_cols as usize;
    let coset = Fp::from(7u64);

    let input: Vec<u64> = (0..total as u64)
        .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .collect();

    // --- Bowers setup ---
    let cols_buf = state.alloc_buffer_data(&input);
    let mut out_buf = state.alloc_buffer::<u64>(total);
    let bowers_tw = cached_bowers_twiddles_goldilocks(log_n);
    let bowers_tw_u64: Vec<u64> = bowers_tw.iter().map(|x| *x.value()).collect();
    let bowers_tw_buf = state.alloc_buffer_data(&bowers_tw_u64);
    let coset_powers = cached_coset_powers_goldilocks(log_n, coset);
    let mut coset_u64: Vec<u64> = coset_powers.iter().map(|x| *x.value()).collect();
    bit_reverse(&mut coset_u64);
    let coset_buf = state.alloc_buffer_data(&coset_u64);

    // --- Baseline setup: radix-2 DIT twiddles + one device buffer per column ---
    let dit_twiddles =
        get_twiddles::<Goldilocks64Field>(log_n as u64, RootsConfig::BitReverse).expect("twiddles");
    let dit_tw_u64: Vec<u64> = dit_twiddles.iter().map(|x| *x.value()).collect();
    let dit_tw_buf = state.alloc_buffer_data(&dit_tw_u64);
    let column_buffers: Vec<_> = (0..num_cols as usize)
        .map(|c| state.alloc_buffer_data(&input[c * n..(c + 1) * n]))
        .collect();

    let mut group = c.benchmark_group("bowers_vs_baseline_2pow20x64");
    group.throughput(Throughput::Elements(total as u64));

    group.bench_function(BenchmarkId::new("bowers_lde", "fused_1_dispatch"), |b| {
        b.iter(|| {
            metal_bowers_lde(
                &cols_buf,
                &bowers_tw_buf,
                &coset_buf,
                &mut out_buf,
                log_n,
                num_cols,
                &state,
            )
            .unwrap();
        });
    });

    group.bench_function(BenchmarkId::new("baseline_dit", "fft_per_column"), |b| {
        b.iter(|| {
            for col_buf in &column_buffers {
                let r = fft_buffer_to_buffer::<Goldilocks64Field>(col_buf, n, &dit_tw_buf, &state)
                    .unwrap();
                black_box(r);
            }
        });
    });

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_bowers_lde, bench_bowers_vs_baseline
);
criterion_main!(benches);
