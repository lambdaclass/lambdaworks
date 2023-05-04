mod functions;
mod util;

fn simple_fibonacci_benchmarks() {
    functions::stark::prove_fib(8);
}

fn two_col_fibonacci_benchmarks() {
    functions::stark::prove_fib_2_cols();
}

fn fibonacci_f17_benchmarks() {
    functions::stark::prove_fib17();
}

fn quadratic_air_benchmarks() {
    functions::stark::prove_quadratic();
}

iai::main!(
    simple_fibonacci_benchmarks,
    two_col_fibonacci_benchmarks,
    fibonacci_f17_benchmarks,
    quadratic_air_benchmarks,
);
