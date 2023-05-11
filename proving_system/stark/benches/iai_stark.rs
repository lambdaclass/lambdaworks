mod functions;
mod util;

#[inline(never)]
fn simple_fibonacci_benchmarks() {
    functions::stark::prove_fib(8);
}

#[inline(never)]
fn two_col_fibonacci_benchmarks() {
    functions::stark::prove_fib_2_cols();
}

#[inline(never)]
fn fibonacci_f17_benchmarks() {
    functions::stark::prove_fib17();
}

#[inline(never)]
fn quadratic_air_benchmarks() {
    functions::stark::prove_quadratic();
}

iai::main!(
    simple_fibonacci_benchmarks,
    two_col_fibonacci_benchmarks,
    fibonacci_f17_benchmarks,
    quadratic_air_benchmarks,
);
