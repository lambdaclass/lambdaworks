use std::hint::black_box;

use criterion::Criterion;
use lambdaworks_math::field::{
    element::FieldElement,
    fields::mersenne31::field::Mersenne31Field,
};
use rand::random;

pub type F = FieldElement<Mersenne31Field>;

#[inline(never)]
#[no_mangle]
#[export_name = "util::rand_field_elements"]
pub fn rand_field_elements(num: usize) -> Vec<(F, F)> {
    let mut result = Vec::with_capacity(num);
    for _ in 0..result.capacity() {
        result.push((F::new(random()), F::new(random())));
    }
    result
}

pub fn mersenne31_ops_benchmarks(c: &mut Criterion) {
    let input: Vec<Vec<(F, F)>> = [1, 10, 100, 1000, 10000, 100000, 1000000]
        .into_iter()
        .map(rand_field_elements)
        .collect::<Vec<_>>();
    let mut group = c.benchmark_group("Mersenne31 operations");

    for i in input.clone().into_iter() {
        group.bench_with_input(format!("add {:?}", &i.len()), &i, |bench, i| {
            bench.iter(|| {
                for (x, y) in i {
                    black_box(black_box(x) + black_box(y));
                }
            });
        });
    }

    for i in input.clone().into_iter() {
        group.bench_with_input(format!("mul {:?}", &i.len()), &i, |bench, i| {
            bench.iter(|| {
                for (x, y) in i {
                    black_box(black_box(x) * black_box(y));
                }
            });
        });
    }

    for i in input.clone().into_iter() {
        group.bench_with_input(format!("pow by 1 {:?}", &i.len()), &i, |bench, i| {
            bench.iter(|| {
                for (x, _) in i {
                    black_box(black_box(x).pow(1_u64));
                }
            });
        });
    }

    for i in input.clone().into_iter() {
        group.bench_with_input(format!("square {:?}", &i.len()), &i, |bench, i| {
            bench.iter(|| {
                for (x, _) in i {
                    black_box(black_box(x).square());
                }
            });
        });
    }

    for i in input.clone().into_iter() {
        group.bench_with_input(format!("square with pow {:?}", &i.len()), &i, |bench, i| {
            bench.iter(|| {
                for (x, _) in i {
                    black_box(black_box(x).pow(2_u64));
                }
            });
        });
    }

    for i in input.clone().into_iter() {
        group.bench_with_input(format!("square with mul {:?}", &i.len()), &i, |bench, i| {
            bench.iter(|| {
                for (x, _) in i {
                    black_box(black_box(x) * black_box(x));
                }
            });
        });
    }

    for i in input.clone().into_iter() {
        group.bench_with_input(
            format!("pow {:?}", &i.len()),
            &(i, 5u64),
            |bench, (i, a)| {
                bench.iter(|| {
                    for (x, _) in i {
                        black_box(black_box(x).pow(*a));
                    }
                });
            },
        );
    }

    for i in input.clone().into_iter() {
        group.bench_with_input(format!("sub {:?}", &i.len()), &i, |bench, i| {
            bench.iter(|| {
                for (x, y) in i {
                    black_box(black_box(x) - black_box(y));
                }
            });
        });
    }

    for i in input.clone().into_iter() {
        group.bench_with_input(format!("inv {:?}", &i.len()), &i, |bench, i| {
            bench.iter(|| {
                for (x, _) in i {
                    black_box(black_box(x).inv().unwrap());
                }
            });
        });
    }

    for i in input.clone().into_iter() {
        group.bench_with_input(format!("div {:?}", &i.len()), &i, |bench, i| {
            bench.iter(|| {
                for (x, y) in i {
                    black_box(black_box(x) / black_box(y));
                }
            });
        });
    }

    for i in input.clone().into_iter() {
        group.bench_with_input(format!("eq {:?}", &i.len()), &i, |bench, i| {
            bench.iter(|| {
                for (x, y) in i {
                    black_box(black_box(x) == black_box(y));
                }
            });
        });
    }

    for i in input.clone().into_iter() {
        group.bench_with_input(format!("sqrt {:?}", &i.len()), &i, |bench, i| {
            bench.iter(|| {
                for (x, _) in i {
                    black_box(black_box(x).sqrt());
                }
            });
        });
    }

    for i in input.clone().into_iter() {
        group.bench_with_input(format!("sqrt squared {:?}", &i.len()), &i, |bench, i| {
            let i: Vec<F> = i.iter().map(|(x, _)| x * x).collect();
            bench.iter(|| {
                for x in &i {
                    black_box(black_box(x).sqrt());
                }
            });
        });
    }

    for i in input.clone().into_iter() {
        group.bench_with_input(format!("bitand {:?}", &i.len()), &i, |bench, i| {
            // Note: we should strive to have the number of limbs be generic... ideally this benchmark group itself should have a generic type that we call into from the main runner.
            let i: Vec<(u32, u32)> =
                i.iter().map(|(x, y)| (*x.value(), *y.value())).collect();
            bench.iter(|| {
                for (x, y) in &i {
                    black_box(black_box(*x) & black_box(*y));
                }
            });
        });
    }

    for i in input.clone().into_iter() {
        group.bench_with_input(format!("bitor {:?}", &i.len()), &i, |bench, i| {
            let i: Vec<(u32, u32)> =
                i.iter().map(|(x, y)| (*x.value(), *y.value())).collect();
            bench.iter(|| {
                for (x, y) in &i {
                    black_box(black_box(*x) | black_box(*y));
                }
            });
        });
    }

    for i in input.clone().into_iter() {
        group.bench_with_input(format!("bitxor {:?}", &i.len()), &i, |bench, i| {
            let i: Vec<(u32, u32)> =
                i.iter().map(|(x, y)| (*x.value(), *y.value())).collect();
            bench.iter(|| {
                for (x, y) in &i {
                    black_box(black_box(*x) ^ black_box(*y));
                }
            });
        });
    }
}
