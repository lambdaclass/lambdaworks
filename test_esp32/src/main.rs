//use esp_idf_hal::peripherals::Peripherals;
//use esp_idf_hal::gpio::PinDriver;
use esp_idf_sys as _;
use lambdaworks_math::field::{
    element::FieldElement,
    fields::montgomery_backed_prime_fields::{IsModulus, U256PrimeField},
    fields::u64_prime_field::U64PrimeField,
    traits::IsTwoAdicField,
};
use lambdaworks_math::polynomial::Polynomial;
use lambdaworks_math::unsigned_integer::element::{UnsignedInteger, U256};
use std::thread;
use std::time::Duration;

use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::MontgomeryConfigStark252PrimeField;
use lambdaworks_stark::air::context::AirContext;
use lambdaworks_stark::air::context::ProofOptions;
use lambdaworks_stark::air::example::simple_fibonacci::FibonacciAIR;
use lambdaworks_stark::air::example::{
    fibonacci_2_columns, fibonacci_f17, quadratic_air, simple_fibonacci,
};
use lambdaworks_stark::air::trace::TraceTable;
use lambdaworks_stark::air::AIR;
use lambdaworks_stark::prover::prove;
use lambdaworks_stark::verifier::verify;

pub type Stark252PrimeField = U256PrimeField<MontgomeryConfigStark252PrimeField>;
type FE = FieldElement<Stark252PrimeField>;

fn main() {
    unsafe {
        // Disable IDLE task WatchDogTask on this CPU.
        esp_idf_sys::esp_task_wdt_delete(esp_idf_sys::xTaskGetIdleTaskHandleForCPU(
            esp_idf_hal::cpu::core() as u32,
        ));

        // Enable WatchDogTask on the main (=this) task.
        //esp_idf_sys::esp_task_wdt_add(esp_idf_sys::xTaskGetCurrentTaskHandle());
    };
    
    let mut i = 0_u64;

    loop {

        test_prove_fib();
/*
        let builder = thread::Builder::new().stack_size(280 * 1024);
        let th = builder.spawn(move || {
            println!("New thread! {i}");

            println!("Test stark! 32");
            test_prove_fib();
        }).unwrap();
*/

        i += 1;

        //th.join().unwrap();

        //thread::sleep(Duration::from_millis(1000));
    }
}

fn test_prove_fib() {
    let trace = simple_fibonacci::fibonacci_trace([FE::from(1), FE::from(1)], 32);
    let trace_length = trace[0].len();
    let trace_table = TraceTable::new_from_cols(&trace);

    let context = AirContext {
        options: ProofOptions {
            blowup_factor: 2,
            fri_number_of_queries: 1,
            coset_offset: 3,
        },
        trace_length,
        trace_columns: trace_table.n_cols,
        transition_degrees: vec![1],
        transition_exemptions: vec![2],
        transition_offsets: vec![0, 1, 2],
        num_transition_constraints: 1,
    };

    let fibonacci_air = FibonacciAIR::new(context);

    let result = prove(&trace_table, &fibonacci_air);
    let cant_query_list = result.query_list.len();
    println!("cant: {cant_query_list}");
    let ret_verify = verify(&result, &fibonacci_air);
    println!("ret_verify: {ret_verify:?}");
}

fn interpolation_example() {
    let elem_one = FE::one();
    let mut val = FE::zero();

    loop {
        let p = Polynomial::interpolate(&[FE::zero(), FE::one()], &[val.clone(), FE::one()]);
        println!("Test lambdaworks!");
        thread::sleep(Duration::from_millis(1000));
        val = val + &elem_one;
        println!("{val:?}");
        let eval = p.evaluate(&val);
        println!("eval: {eval:?}");
    }
}
