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

iai_callgrind::main!(
    callgrind_args = "toggle-collect=util::*";
    functions = simple_fibonacci_benchmarks, two_col_fibonacci_benchmarks,
);
