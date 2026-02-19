use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use lambdaworks_crypto::hash::rescue::{Fp, MdsMethod, Rpo256, Rpx256, SecurityLevel};
use rand::{Rng, SeedableRng};

fn generate_random_input(size: usize) -> Vec<Fp> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    (0..size).map(|_| Fp::from(rng.gen::<u64>())).collect()
}

fn generate_random_bytes(size: usize) -> Vec<u8> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    (0..size).map(|_| rng.gen::<u8>()).collect()
}

fn rescue_benchmarks(c: &mut Criterion) {
    let rpo = Rpo256::new(SecurityLevel::Sec128, MdsMethod::MatrixMultiplication).unwrap();
    let rpx = Rpx256::new(MdsMethod::MatrixMultiplication).unwrap();

    let input_8 = generate_random_input(8);
    let input_64 = generate_random_input(64);
    let input_256 = generate_random_input(256);
    let input_1024 = generate_random_input(1024);

    let bytes_64 = generate_random_bytes(64);
    let bytes_256 = generate_random_bytes(256);
    let bytes_1024 = generate_random_bytes(1024);

    // Hash elements benchmarks
    let mut group = c.benchmark_group("hash_elements");

    group.throughput(Throughput::Elements(8));
    group.bench_function("RPO_8_elements", |b| {
        b.iter(|| black_box(rpo.hash(&input_8)))
    });
    group.bench_function("RPX_8_elements", |b| {
        b.iter(|| black_box(rpx.hash(&input_8)))
    });

    group.throughput(Throughput::Elements(64));
    group.bench_function("RPO_64_elements", |b| {
        b.iter(|| black_box(rpo.hash(&input_64)))
    });
    group.bench_function("RPX_64_elements", |b| {
        b.iter(|| black_box(rpx.hash(&input_64)))
    });

    group.throughput(Throughput::Elements(256));
    group.bench_function("RPO_256_elements", |b| {
        b.iter(|| black_box(rpo.hash(&input_256)))
    });
    group.bench_function("RPX_256_elements", |b| {
        b.iter(|| black_box(rpx.hash(&input_256)))
    });

    group.throughput(Throughput::Elements(1024));
    group.bench_function("RPO_1024_elements", |b| {
        b.iter(|| black_box(rpo.hash(&input_1024)))
    });
    group.bench_function("RPX_1024_elements", |b| {
        b.iter(|| black_box(rpx.hash(&input_1024)))
    });

    group.finish();

    // Hash bytes benchmarks
    let mut group = c.benchmark_group("hash_bytes");

    group.throughput(Throughput::Bytes(64));
    group.bench_function("RPO_64_bytes", |b| {
        b.iter(|| black_box(rpo.hash_bytes(&bytes_64)))
    });
    group.bench_function("RPX_64_bytes", |b| {
        b.iter(|| black_box(rpx.hash_bytes(&bytes_64)))
    });

    group.throughput(Throughput::Bytes(256));
    group.bench_function("RPO_256_bytes", |b| {
        b.iter(|| black_box(rpo.hash_bytes(&bytes_256)))
    });
    group.bench_function("RPX_256_bytes", |b| {
        b.iter(|| black_box(rpx.hash_bytes(&bytes_256)))
    });

    group.throughput(Throughput::Bytes(1024));
    group.bench_function("RPO_1024_bytes", |b| {
        b.iter(|| black_box(rpo.hash_bytes(&bytes_1024)))
    });
    group.bench_function("RPX_1024_bytes", |b| {
        b.iter(|| black_box(rpx.hash_bytes(&bytes_1024)))
    });

    group.finish();

    // Permutation only benchmarks
    let mut group = c.benchmark_group("permutation");

    let mut state_rpo = input_8.clone();
    let mut state_rpx = input_8.clone();
    state_rpo.resize(12, Fp::zero());
    state_rpx.resize(12, Fp::zero());

    group.bench_function("RPO_permutation", |b| {
        b.iter(|| {
            let mut s = state_rpo.clone();
            rpo.permutation(&mut s);
            black_box(s)
        })
    });
    group.bench_function("RPX_permutation", |b| {
        b.iter(|| {
            let mut s = state_rpx.clone();
            rpx.permutation(&mut s);
            black_box(s)
        })
    });

    group.finish();

    // MDS method comparison for RPX
    let rpo_ntt = Rpo256::new(SecurityLevel::Sec128, MdsMethod::Ntt).unwrap();
    let rpo_karatsuba = Rpo256::new(SecurityLevel::Sec128, MdsMethod::Karatsuba).unwrap();
    let rpx_ntt = Rpx256::new(MdsMethod::Ntt).unwrap();
    let rpx_karatsuba = Rpx256::new(MdsMethod::Karatsuba).unwrap();

    let mut group = c.benchmark_group("mds_methods_64_elements");
    group.throughput(Throughput::Elements(64));

    group.bench_function("RPO_Matrix", |b| b.iter(|| black_box(rpo.hash(&input_64))));
    group.bench_function("RPO_NTT", |b| b.iter(|| black_box(rpo_ntt.hash(&input_64))));
    group.bench_function("RPO_Karatsuba", |b| {
        b.iter(|| black_box(rpo_karatsuba.hash(&input_64)))
    });

    group.bench_function("RPX_Matrix", |b| b.iter(|| black_box(rpx.hash(&input_64))));
    group.bench_function("RPX_NTT", |b| b.iter(|| black_box(rpx_ntt.hash(&input_64))));
    group.bench_function("RPX_Karatsuba", |b| {
        b.iter(|| black_box(rpx_karatsuba.hash(&input_64)))
    });

    group.finish();
}

criterion_group!(rescue, rescue_benchmarks);
criterion_main!(rescue);
