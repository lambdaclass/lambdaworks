//! Criterion benchmark for the Metal Bowers-G fused NTT + coset LDE.
//!
//! Measures `metal_bowers_lde` throughput at STARK-prover matrix shapes.
//! Run with: `cargo bench -p lambdaworks-math --features metal --bench criterion_metal_bowers_lde`

#![cfg(all(target_os = "macos", feature = "metal"))]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use lambdaworks_gpu::metal::abstractions::state::MetalState;
use lambdaworks_math::fft::gpu::metal::bowers::dispatcher::metal_bowers_lde;
use lambdaworks_math::fft::gpu::metal::bowers::twiddles::{
    cached_bowers_twiddles_goldilocks, cached_coset_powers_goldilocks,
};
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;

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

    // (log_n, num_cols): the target STARK shape and a couple of neighbours.
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

criterion_group!(benches, bench_bowers_lde);
criterion_main!(benches);
