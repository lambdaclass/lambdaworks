use criterion::{criterion_group, criterion_main, Criterion};
use pprof::criterion::{Output, PProfProfiler};

mod elliptic_curves;
use elliptic_curves::{
    bls12_377::bls12377_elliptic_curve_benchmarks, bls12_381::bls12381_elliptic_curve_benchmarks,
};

//TODO: add multiple bench targets per curve with a main `elliptic_curve_target` this will allow us to run individual and complete benches
//WAIT! Since criterion performs keyword search perhaps the verbose but correct pattern is to include individual groups for all benches and just key word search..
//however this makes it incompatible with processing over the same seedable input..
//WAIT! If its seedable all we have to do is run all benches and they will be done over the same random input.
criterion_group!(
    name = elliptic_curve_benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets =  bls12381_elliptic_curve_benchmarks, bls12377_elliptic_curve_benchmarks
);
criterion_main!(elliptic_curve_benches);
