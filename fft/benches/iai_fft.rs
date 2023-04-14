use iai::black_box;
use lambdaworks_fft::roots_of_unity::get_twiddles;
use lambdaworks_math::field::traits::RootsConfig;

mod functions;
mod util;

const SIZE_ORDERS: [u64; 4] = [21, 22, 23, 24];

pub fn fft_benchmarks() {
    for order in SIZE_ORDERS {
        let input = util::rand_field_elements(order);
        let twiddles_bitrev = get_twiddles(order, RootsConfig::BitReverse).unwrap();
        let twiddles_nat = get_twiddles(order, RootsConfig::Natural).unwrap();

        functions::ordered_fft_nr(black_box(&input), black_box(&twiddles_bitrev));
        functions::ordered_fft_rn(black_box(&input), black_box(&twiddles_nat));
    }
}

fn twiddles_generation_benchmarks() {
    for order in SIZE_ORDERS {
        functions::twiddles_generation(black_box(order));
    }
}

fn bitrev_permutation_benchmarks() {
    for input in SIZE_ORDERS.map(util::rand_field_elements) {
        functions::bitrev_permute(black_box(&input));
    }
}

fn poly_evaluation_benchmarks() {
    for poly in SIZE_ORDERS.map(util::rand_poly) {
        functions::poly_evaluate_fft(black_box(&poly));
    }
}

fn poly_interpolation_benchmarks() {
    for evals in SIZE_ORDERS.map(util::rand_field_elements) {
        functions::poly_interpolate_fft(black_box(&evals));
    }
}

iai::main!(
    fft_benchmarks,
    twiddles_generation_benchmarks,
    bitrev_permutation_benchmarks,
    poly_evaluation_benchmarks,
    poly_interpolation_benchmarks
);
