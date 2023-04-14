mod functions;
mod util;

fn fft_benchmarks() {
    for input in SIZE_ORDERS.map(util::rand_field_elements) {
        let metal_state = MetalState::new(None).unwrap();
        let twiddles = gen_twiddles::<F>(order, RootsConfig::BitReverse, &metal_state).unwrap();

        functions::metal::ordered_fft(&input, &twiddles);
    }
}
