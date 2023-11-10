use criterion::{black_box, Criterion};
use lambdaworks_math::{
    cyclic_group::IsGroup,
    elliptic_curve::{
        short_weierstrass::{
            curves::bls12_381::{
                compression::{check_point_is_in_subgroup, compress_g1_point, decompress_g1_point},
                curve::BLS12381Curve,
                pairing::BLS12381AtePairing,
                twist::BLS12381TwistCurve,
            },
            point::ShortWeierstrassProjectivePoint,
        },
        traits::{IsEllipticCurve, IsPairing},
    },
};
use rand::{rngs::StdRng, Rng, SeedableRng};

type G1 = ShortWeierstrassProjectivePoint<BLS12381Curve>;
type G2 = ShortWeierstrassProjectivePoint<BLS12381TwistCurve>;

fn rand_pairing_points(num: usize) -> ((Vec<G1>, Vec<G2>), (Vec<u128>, Vec<u128>)) {
    let mut rng = StdRng::seed_from_u64(42);
    let mut g1s = Vec::with_capacity(num);
    let mut g2s = Vec::with_capacity(num);
    let mut a_vals = Vec::with_capacity(num);
    let mut b_vals = Vec::with_capacity(num);

    for _ in 0..num {
        let g1 = BLS12381Curve::generator();
        let g2 = BLS12381TwistCurve::generator();
        let a: u128 = rng.gen();
        let b: u128 = rng.gen();
        g1s.push(g1.operate_with_self(a));
        g2s.push(g2.operate_with_self(b));
        a_vals.push(a);
        b_vals.push(b);
    }
    ((g1s, g2s), (a_vals, b_vals))
}

pub fn bls12381_elliptic_curve_benchmarks(c: &mut Criterion) {
    //TODO: find a better method of generating and representing benchmark data. This is very verbose and redunant.
    let input: Vec<((Vec<G1>, Vec<G2>), (Vec<u128>, Vec<u128>))> =
        [1, 10, 100, 1000, 10000, 100000, 1000000]
            .into_iter()
            .map(rand_pairing_points)
            .collect::<Vec<_>>();
    let operate_with_points = input
        .iter()
        .fold(Vec::new(), |mut acc, ((g1, g2), (a, b))| {
            let g1s = g1
                .iter()
                .zip(b.iter())
                .map(|(g1, b)| g1.operate_with_self(*b))
                .collect::<Vec<_>>();
            let g2s = g2
                .iter()
                .zip(a.iter())
                .map(|(g2, a)| g2.operate_with_self(*a))
                .collect::<Vec<_>>();
            acc.push((g1s, g2s));
            acc
        });

    let mut group = c.benchmark_group("BLS12-381 Ops");

    // Operate_with G1
    for (((g1, _), _), (g1_b, _)) in input
        .clone()
        .into_iter()
        .zip(operate_with_points.clone().into_iter())
    {
        group.bench_with_input(
            format!("Operate_with_G1 {:?}", &g1.len()),
            &(g1, g1_b),
            |b, (g1, g1_b)| {
                b.iter(|| {
                    for (p, b) in g1.iter().zip(g1_b.iter()) {
                        black_box(black_box(p).operate_with(black_box(b)));
                    }
                });
            },
        );
    }

    // Operate_with G2
    for (((_, g2), _), (_, g2_a)) in input
        .clone()
        .into_iter()
        .zip(operate_with_points.clone().into_iter())
    {
        group.bench_with_input(
            format!("Operate_with_G2 {:?}", &g2.len()),
            &(g2, g2_a),
            |b, (g2, g2_a)| {
                b.iter(|| {
                    for (p, a) in g2.iter().zip(g2_a.iter()) {
                        black_box(black_box(p).operate_with(black_box(a)));
                    }
                });
            },
        );
    }

    // Operate_with_self G1
    for ((g1, _), (_, b)) in input.clone().into_iter() {
        group.bench_with_input(
            format!("Operate_with_self_G1 {:?}", &g1.len()),
            &(g1, b),
            |ben, (g1, b)| {
                ben.iter(|| {
                    for (p, b) in g1.iter().zip(b.iter()) {
                        black_box(black_box(p).operate_with_self(black_box(*b)));
                    }
                });
            },
        );
    }

    // Operate_with_self G2
    for ((_, g2), (_, a)) in input.clone().into_iter() {
        group.bench_with_input(
            format!("Operate_with_self_G2 {:?}", &g2.len()),
            &(g2, a),
            |b, (g2, a)| {
                b.iter(|| {
                    for (p, a) in g2.iter().zip(a.iter()) {
                        black_box(black_box(p).operate_with_self(black_box(*a)));
                    }
                });
            },
        );
    }

    // Double G1
    for ((g1, _), _) in input.clone().into_iter() {
        group.bench_with_input(format!("Double G1 {:?}", &g1.len()), &g1, |b, g1| {
            b.iter(|| {
                for p in g1 {
                    black_box(black_box(p).operate_with_self(black_box(2u64)));
                }
            });
        });
    }

    // Double G2
    for ((_, g2), _) in input.clone().into_iter() {
        group.bench_with_input(format!("Double G2 {:?}", &g2.len()), &g2, |b, g2| {
            b.iter(|| {
                for p in g2 {
                    black_box(black_box(p).operate_with_self(black_box(2u64)));
                }
            });
        });
    }

    // Neg G1
    for ((g1, _), _) in input.clone().into_iter() {
        group.bench_with_input(format!("Neg G1 {:?}", &g1.len()), &g1, |b, g1| {
            b.iter(|| {
                for p in g1 {
                    black_box(black_box(p).neg());
                }
            });
        });
    }

    // Neg G2
    for ((_, g2), _) in input.clone().into_iter() {
        group.bench_with_input(format!("Neg G2 {:?}", &g2.len()), &g2, |b, g2| {
            b.iter(|| {
                for p in g2 {
                    black_box(black_box(p).neg());
                }
            });
        });
    }

    // Compress_G1_point
    for ((g1, _), _) in input.clone().into_iter() {
        group.bench_with_input(
            format!("Compress G1 point {:?}", &g1.len()),
            &g1,
            |b, g1| {
                b.iter(|| {
                    for p in g1 {
                        black_box(compress_g1_point(black_box(p)));
                    }
                });
            },
        );
    }

    // Decompress_G1_point
    for ((g1, _), _) in input.clone().into_iter() {
        group.bench_with_input(
            format!("Decompress G1 Point {:?}", &g1.len()),
            &g1,
            |b, g1| {
                let g1_data = g1
                    .into_iter()
                    .map(|g1| {
                        //Eww
                        let a: [u8; 48] = compress_g1_point(g1).try_into().unwrap();
                        a
                    })
                    .collect::<Vec<_>>();
                b.iter(|| {
                    for p in g1_data.clone() {
                        //Rust Q: does calling unwrap() here effect the benchmark
                        black_box(decompress_g1_point(&mut black_box(p))).unwrap();
                    }
                });
            },
        );
    }

    // Subgroup Check
    for ((g1, _), _) in input.clone().into_iter() {
        group.bench_with_input(format!("Subgroup Check {:?}", &g1.len()), &g1, |b, g1| {
            b.iter(|| {
                for p in g1 {
                    black_box(check_point_is_in_subgroup(black_box(p)));
                }
            });
        });
    }

    // Ate Pairing
    for ((g1, g2), _) in input.clone().into_iter() {
        group.bench_with_input(
            format!("Ate Pairing {:?}", &g1.len()),
            &(g1, g2),
            |b, (g1, g2)| {
                b.iter(|| {
                    for (p, q) in g1.iter().zip(g2.iter()) {
                        black_box(BLS12381AtePairing::compute(black_box(&p), black_box(&q)));
                    }
                });
            },
        );
    }
}
