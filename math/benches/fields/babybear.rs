use criterion::Criterion;
use std::hint::black_box;

use lambdaworks_math::field::fields::fft_friendly::quadratic_babybear::QuadraticBabybearField;
use lambdaworks_math::field::fields::fft_friendly::quartic_babybear::Degree4BabyBearExtensionField;
use lambdaworks_math::field::{
    element::FieldElement,
    errors::FieldError,
    fields::fft_friendly::babybear::Babybear31PrimeField,
    traits::{IsFFTField, IsField, IsSubFieldOf},
};

use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, FieldAlgebra};

use rand::random;

use rand::Rng;

pub type F = FieldElement<Babybear31PrimeField>;
pub type Fp2E = FieldElement<QuadraticBabybearField>;
pub type Fp4E = FieldElement<Degree4BabyBearExtensionField>;
type EF4 = BinomialExtensionField<BabyBear, 4>;

// Create a vector of random field elements for the elements using LambdaWorks

pub fn rand_field_elements(num: usize) -> Vec<(F, F)> {
    //let mut result = Vec::with_capacity(num);
    let mut result = Vec::with_capacity(num);
    for _ in 0..result.capacity() {
        result.push((F::from(random::<u64>()), F::from(random::<u64>())));
    }
    result
}

pub fn rand_babybear_fp4_elements(num: usize) -> Vec<(Fp4E, Fp4E)> {
    let mut result = Vec::with_capacity(num);
    for _ in 0..num {
        result.push((
            Fp4E::new([
                F::from(random::<u64>()),
                F::from(random::<u64>()),
                F::from(random::<u64>()),
                F::from(random::<u64>()),
            ]),
            Fp4E::new([
                F::from(random::<u64>()),
                F::from(random::<u64>()),
                F::from(random::<u64>()),
                F::from(random::<u64>()),
            ]),
        ));
    }
    result
}

// Create a vector of random field elements for the elements using Plonky3
// use u64?

//to do create u32 for montgomery in lambdaworks?
// use a more idiomatic way to do the benches

fn rand_babybear_elements_p3(num: usize) -> Vec<(BabyBear, BabyBear)> {
    let mut rng = rand::thread_rng();
    (0..num)
        .map(|_| (rng.gen::<BabyBear>(), rng.gen::<BabyBear>()))
        .collect()
}

fn rand_babybear_fp4_elements_p3(num: usize) -> Vec<(EF4, EF4)> {
    let mut rng = rand::thread_rng();
    (0..num)
        .map(|_| (rng.gen::<EF4>(), rng.gen::<EF4>()))
        .collect()
}

// Operations for BabyBear extension field  4 using Lambdaworks
pub fn babybear_extension_ops_benchmarks(c: &mut Criterion) {
    let input: Vec<Vec<(Fp4E, Fp4E)>> = [1, 10, 100, 1000, 10000, 100000, 1000000]
        .into_iter()
        .map(rand_babybear_fp4_elements)
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("BabyBear Fp4 operations");

    for i in input.clone().into_iter() {
        group.bench_with_input(format!("Add of Fp4 {:?}", &i.len()), &i, |bench, i| {
            bench.iter(|| {
                for (x, y) in i {
                    black_box(black_box(x) + black_box(y));
                }
            });
        });
    }

    for i in input.clone().into_iter() {
        group.bench_with_input(format!("Mul of Fp4 {:?}", &i.len()), &i, |bench, i| {
            bench.iter(|| {
                for (x, y) in i {
                    black_box(black_box(x) * black_box(y));
                }
            });
        });
    }

    for i in input.clone().into_iter() {
        group.bench_with_input(format!("Square of Fp4 {:?}", &i.len()), &i, |bench, i| {
            bench.iter(|| {
                for (x, _) in i {
                    black_box(black_box(x).square());
                }
            });
        });
    }

    for i in input.clone().into_iter() {
        group.bench_with_input(format!("Inv of Fp4 {:?}", &i.len()), &i, |bench, i| {
            bench.iter(|| {
                for (x, y) in i {
                    black_box(black_box(x) / black_box(y));
                }
            });
        });
    }

    for i in input.clone().into_iter() {
        group.bench_with_input(format!("Div of Fp4 {:?}", &i.len()), &i, |bench, i| {
            bench.iter(|| {
                for (x, _) in i {
                    black_box(black_box(x).inv().unwrap());
                }
            });
        });
    }
}

pub fn babybear_ops_benchmarks(c: &mut Criterion) {
    let input: Vec<Vec<(F, F)>> = [1, 10, 100, 1000, 10000, 100000, 1000000]
        .into_iter()
        .map(rand_field_elements)
        .collect::<Vec<_>>();
    let mut group = c.benchmark_group("BabyBear operations using Lambdaworks");

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
        group.bench_with_input(format!("square {:?}", &i.len()), &i, |bench, i| {
            bench.iter(|| {
                for (x, _) in i {
                    black_box(black_box(x).square());
                }
            });
        });
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
}
// Operations benchmarks for BabyBear field using Plonky3
pub fn babybear_p3_ops_benchmarks(c: &mut Criterion) {
    let input: Vec<Vec<(BabyBear, BabyBear)>> = [1, 10, 100, 1000, 10000, 100000, 1000000]
        .into_iter()
        .map(rand_babybear_elements_p3)
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("BabyBear operations using Plonky3");

    for i in input.clone().into_iter() {
        group.bench_with_input(format!("add {:?}", &i.len()), &i, |bench, i| {
            bench.iter(|| {
                for (x, y) in i {
                    black_box(black_box(*x) + black_box(*y));
                }
            });
        });
    }

    for i in input.clone().into_iter() {
        group.bench_with_input(format!("sub {:?}", &i.len()), &i, |bench, i| {
            bench.iter(|| {
                for (x, y) in i {
                    black_box(black_box(*x) - black_box(*y));
                }
            });
        });
    }
    for i in input.clone().into_iter() {
        group.bench_with_input(format!("mul {:?}", &i.len()), &i, |bench, i| {
            bench.iter(|| {
                for (x, y) in i {
                    black_box(black_box(*x) * black_box(*y));
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
        group.bench_with_input(format!("inv {:?}", &i.len()), &i, |bench, i| {
            bench.iter(|| {
                for (x, _) in i {
                    black_box(black_box(x).inverse());
                }
            });
        });
    }

    for i in input.clone().into_iter() {
        group.bench_with_input(format!("div {:?}", &i.len()), &i, |bench, i| {
            bench.iter(|| {
                for (x, y) in i {
                    black_box(black_box(*x) / black_box(*y));
                }
            });
        });
    }
}

// Operations benchmarks for BabyBear extension 4 field  using Plonky3

pub fn babybear_extension_ops_benchmarks_p3(c: &mut Criterion) {
    let input_sizes = [1, 10, 100, 1000, 10000, 100000, 1000000];
    let input: Vec<Vec<(EF4, EF4)>> = input_sizes
        .into_iter()
        .map(|size| rand_babybear_fp4_elements_p3(size))
        .collect::<Vec<_>>();

    let mut group = c.benchmark_group("BabyBear Fp4 operations using Plonky3");

    for i in input.clone().into_iter() {
        group.bench_with_input(format!("Add of Fp4 {:?}", &i.len()), &i, |bench, i| {
            bench.iter(|| {
                for (x, y) in i {
                    black_box(black_box(*x) + black_box(*y));
                }
            });
        });
    }
    for i in input.clone().into_iter() {
        group.bench_with_input(format!("Mul of Fp4 {:?}", &i.len()), &i, |bench, i| {
            bench.iter(|| {
                for (x, y) in i {
                    black_box(black_box(*x) * black_box(*y));
                }
            });
        });
    }
    for i in input.clone().into_iter() {
        group.bench_with_input(format!("Square of Fp4 {:?}", &i.len()), &i, |bench, i| {
            bench.iter(|| {
                for (x, _) in i {
                    black_box(black_box(x).square());
                }
            });
        });
    }

    for i in input.clone().into_iter() {
        group.bench_with_input(format!("Inv of Fp4 {:?}", &i.len()), &i, |bench, i| {
            bench.iter(|| {
                for (x, _) in i {
                    black_box(black_box(x).inverse());
                }
            });
        });
    }

    for i in input.clone().into_iter() {
        group.bench_with_input(format!("Div of Fp4 {:?}", &i.len()), &i, |bench, i| {
            bench.iter(|| {
                for (x, y) in i {
                    black_box(black_box(*x) / black_box(*y));
                }
            });
        });
    }
}
