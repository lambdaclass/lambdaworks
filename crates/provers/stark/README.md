<div align="center">

# 🌟 Lambdaworks Stark Platinum Prover 🌟

<img src="https://github.com/lambdaclass/lambdaworks_stark_platinum/assets/569014/ad8d7943-f011-49b5-a0c5-f07e5ef4133e" alt="drawing" width="300"/>

## An open-source STARK prover, drop-in replacement for Winterfell.

</div>

[![Telegram Chat][tg-badge]][tg-url]

[tg-badge]: https://img.shields.io/static/v1?color=green&logo=telegram&label=chat&style=flat&message=join
[tg-url]: https://t.me/+98Whlzql7Hs0MDZh

## ⚠️ Disclaimer

This prover is still in development and may contain bugs. It is not intended to be used in production yet. 

## Description

This is a [STARK prover and verifier](https://eprint.iacr.org/2018/046), which is a transparent (no trusted setup) and post-quantum secure argument of knowledge. The main ingredients are:
- [Hash functions](../../crypto/src/hash/README.md)
- [Fiat-Shamir transformation](../../crypto/src/fiat_shamir/README.md)
- [Finite fields](../../math/src/field/README.md)
- [Univariate polynomials](../../math/src/polynomial/README.md)
- [Reed-Solomon codes](https://en.wikipedia.org/wiki/Reed%E2%80%93Solomon_error_correction)

The security of STARKs depends on collision-resistant hash functions. The security level depends on the number of queries and the size of the underlying field. The prover works either with:
- Finite fields of prime order, where the size of the field should be at least 128 bits.
- Field extensions, where the size of the extension should be at least 128 bits.

The field (or base field $\mathbb{F}_p$ in case of extensions $\mathbb{F}_{p^k}$) has to implement the trait `IsFFTField`, ensuring we can use the [FFT algorithm](../../math/src/fft/README.md) (which is crucial for efficiency). Some fields implementing this trait are:
- [STARK-252](../../math/src/field/fields/fft_friendly/stark_252_prime_field.rs)
- [Baby-Bear](../../math/src/field/fields/fft_friendly/babybear_u32.rs) with its [quartic degree extension](../../math/src/field/fields/fft_friendly/quartic_babybear_u32.rs)

To prove a statement, we will need a description of it, in the form of an Algebraic Intermediate Representation (AIR). This consists of:
- One or more tables (trace and auxiliary trace)
- A set of polynomial equations that have to be enforced on the trace (constraints)

## [Documentation](https://lambdaclass.github.io/lambdaworks/starks/cairo.html)

## Examples

You can take a look at the examples for [read-only memory](https://blog.lambdaclass.com/continuous-read-only-memory-constraints-an-implementation-using-lambdaworks/) and [logUp](https://blog.lambdaclass.com/logup-lookup-argument-and-its-implementation-using-lambdaworks-for-continuous-read-only-memory/).

The examples are [here](./src/examples/) and you can take a look at [integration tests](./src/tests/integration_tests.rs).

The following code summarizes the procedure to generate and verify a STARK proof that attests to the validity of the computation of the 8th [Fibonacci number](https://en.wikipedia.org/wiki/Fibonacci_sequence).
```rust
    let mut trace = simple_fibonacci::fibonacci_trace([Felt252::from(1), Felt252::from(1)], 8);

    let proof_options = ProofOptions::default_test_options();

    let pub_inputs = FibonacciPublicInputs {
        a0: Felt252::one(),
        a1: Felt252::one(),
    };

    let proof = Prover::<FibonacciAIR<Stark252PrimeField>>::prove(
        &mut trace,
        &pub_inputs,
        &proof_options,
        StoneProverTranscript::new(&[]),
    )
    .unwrap();
    assert!(Verifier::<FibonacciAIR<Stark252PrimeField>>::verify(
        &proof,
        &pub_inputs,
        &proof_options,
        StoneProverTranscript::new(&[]),
    ));
```
Here we want to compute the following program:
$a_0 = 1$
$a_1 = 1$
$a_{n + 2} = a_{n + 1} + a_n$ for $n \in \{ 0, 1, ... , 6 \}$
We are going to organize the numbers in one single column. The program is valid if and only if:
- $a_0 - 1 = 0$
- $a_1 - 1 = 0$
- $a_{n + 2} - a_{n + 1} - a_n = 0$

The first step
```rust
let mut trace = simple_fibonacci::fibonacci_trace([Felt252::from(1), Felt252::from(1)], 8);
```
generates the table containing the first 8 Fibonacci numbers. This is what the function does under the hood:
```rust
pub fn fibonacci_trace<F: IsFFTField>(
    initial_values: [FieldElement<F>; 2],
    trace_length: usize,
) -> TraceTable<F, F> {
    let mut ret: Vec<FieldElement<F>> = vec![];

    ret.push(initial_values[0].clone());
    ret.push(initial_values[1].clone());

    for i in 2..(trace_length) {
        ret.push(ret[i - 1].clone() + ret[i - 2].clone());
    }

    TraceTable::from_columns_main(vec![ret], 1)
}
```
Notice the field `F` implements the trait `IsFFTField`. Then we set the proof options,
```rust
let proof_options = ProofOptions::default_test_options();
```
Here we set the options for the test to the default. This determine the blow up factor (i.e. the inverse of the rate of the code) number of queries, whether the prover needs to solve a proof of work before sampling queries (grinding) and the coset offset (a field element which should not be in the domain used to interpolate). Since this is a test, the security level is too low, but the user can use other options that ensure the right security level (for example, 128 bits). Except for the offset, the rest influence the speed of the prover, the verification time and the proof size.

We now create the public inputs for the program.
```rust
let pub_inputs = FibonacciPublicInputs {
        a0: Felt252::one(),
        a1: Felt252::one(),
    };
```
These values are known by prover and verifier. They have to be enforced on the trace and are incorporated into the transcript (otherwise you have a weak version of Fiat-Shamir that can be exploited). So far, we have a table (trace) containing all the intermediate steps of the program and the public inputs (here we only care about the initial values, but we could also include the output of the program). Values in the trace which are not public inputs are witness values. Our program will prove that we computed correctly the 8th Fibonacci number, without revealing it (if we wanted to show the number, it must be included among the public input).

To generate the proof, we simply call
```rust
 let proof = Prover::<FibonacciAIR<Stark252PrimeField>>::prove(
        &mut trace,
        &pub_inputs,
        &proof_options,
        StoneProverTranscript::new(&[]),
    )
    .unwrap();
```
The prover has `FibonacciAIR<Stark252PrimeField>` which defines the AIR over the STARK-252 finite field. A closer look shows that
```rust
pub struct FibonacciAIR<F>
where
    F: IsFFTField,
{
    context: AirContext,
    trace_length: usize,
    pub_inputs: FibonacciPublicInputs<F>,
    constraints: Vec<Box<dyn TransitionConstraint<F, F>>>,
}
```
The AIR contains an `AirContext`, the length of the table (if we wanted to prove that we calculated correctly the 16th Fibonacci number, this should change), the public input and a vector of constraints `constraints: Vec<Box<dyn TransitionConstraint<F, F>>>`. The public input is
```rust
pub struct FibonacciPublicInputs<F>
where
    F: IsFFTField,
{
    pub a0: FieldElement<F>,
    pub a1: FieldElement<F>,
}
```
The following code implements the AIR trait for `FibonacciAIR<F>`
```rust
impl<F> AIR for FibonacciAIR<F>
where
    F: IsFFTField + Send + Sync + 'static,
{
    type Field = F;
    type FieldExtension = F;
    type PublicInputs = FibonacciPublicInputs<Self::Field>;

    const STEP_SIZE: usize = 1;

    fn new(
        trace_length: usize,
        pub_inputs: &Self::PublicInputs,
        proof_options: &ProofOptions,
    ) -> Self {
        let constraints: Vec<Box<dyn TransitionConstraint<F, F>>> =
            vec![Box::new(FibConstraint::new())];

        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns: 1,
            transition_offsets: vec![0, 1, 2],
            num_transition_constraints: constraints.len(),
        };

        Self {
            pub_inputs: pub_inputs.clone(),
            context,
            trace_length,
            constraints,
        }
    }

    fn composition_poly_degree_bound(&self) -> usize {
        self.trace_length()
    }

    fn transition_constraints(&self) -> &Vec<Box<dyn TransitionConstraint<F, F>>> {
        &self.constraints
    }

    fn boundary_constraints(
        &self,
        _rap_challenges: &[FieldElement<Self::Field>],
    ) -> BoundaryConstraints<Self::Field> {
        let a0 = BoundaryConstraint::new_simple_main(0, self.pub_inputs.a0.clone());
        let a1 = BoundaryConstraint::new_simple_main(1, self.pub_inputs.a1.clone());

        BoundaryConstraints::from_constraints(vec![a0, a1])
    }

    fn context(&self) -> &AirContext {
        &self.context
    }

    fn trace_length(&self) -> usize {
        self.trace_length
    }

    fn trace_layout(&self) -> (usize, usize) {
        (1, 0)
    }

    fn pub_inputs(&self) -> &Self::PublicInputs {
        &self.pub_inputs
    }
}
```
When defining the AIR, we need to define constraints (either consistency, type or transition constraints) and boundary constraints (these are defined separately, which enforce that a particular public value holds in a specific place of the table). The context contains useful data related to proof options, number of transition constraints, number of columns in the trace and offsets (these are needed when we want to evaluate constraints).

We also have some auxiliary methods that give the bound of the composition polynomial (`composition_poly_degree_bound`), trace layout (`trace_layout`), public input (`pub_inputs`), trace length (`trace_length`) and AIR context (`context`). Below we show the definition of the transition constraint for Fibonacci, which encodes $a_{n + 2} - a_{n + 1} - a_n = 0$
```rust
#[derive(Clone)]
struct FibConstraint<F: IsFFTField> {
    phantom: PhantomData<F>,
}

impl<F: IsFFTField> FibConstraint<F> {
    pub fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<F> TransitionConstraint<F, F> for FibConstraint<F>
where
    F: IsFFTField + Send + Sync,
{
    fn degree(&self) -> usize {
        1
    }

    fn constraint_idx(&self) -> usize {
        0
    }

    fn end_exemptions(&self) -> usize {
        2
    }

    fn evaluate(
        &self,
        evaluation_context: &TransitionEvaluationContext<F, F>,
        transition_evaluations: &mut [FieldElement<F>],
    ) {
        let (frame, _periodic_values, _rap_challenges) = match evaluation_context {
            TransitionEvaluationContext::Prover {
                frame,
                periodic_values,
                rap_challenges,
            }
            | TransitionEvaluationContext::Verifier {
                frame,
                periodic_values,
                rap_challenges,
            } => (frame, periodic_values, rap_challenges),
        };

        let first_step = frame.get_evaluation_step(0);
        let second_step = frame.get_evaluation_step(1);
        let third_step = frame.get_evaluation_step(2);

        let a0 = first_step.get_main_evaluation_element(0, 0);
        let a1 = second_step.get_main_evaluation_element(0, 0);
        let a2 = third_step.get_main_evaluation_element(0, 0);

        let res = a2 - a1 - a0;

        transition_evaluations[self.constraint_idx()] = res;
    }
}
```
The most important method is `evaluate`:
```rust 
 let res = a2 - a1 - a0;
 transition_evaluations[self.constraint_idx()] = res;
```
The function will take three consecutive values of the (low-degree extension of) trace and compute the relation defining the Fibonacci sequence. These values will be later used to reconstruct a polynomial enforcing the constraints. Some important parameters of transition constraints are:
- `degree`: indicates the highest degree in the algebraic expression defining the constraint. For Fibonacci ($a_2 - a_1 - a_0$) is linear, so degree is 1, but if the constraint were $a_2*a_1^2 - a_0^2$, it would have been 3.
- `idx`: only relevant when there are multiple constraints, it assigns a number allowing to identify the constraint more easily.
- `end_exemptions`: indicates where the constraint is enforced. If `0`, then it is valid over the whole length of the trace; `1` means that it is valid over all the trace except the last value, `2` that is valid everywhere except the last two values and so on. This is relevant to compute the zerofiers/vanishing polynomials efficiently.

The boundary constraints are defined as follows:
```rust 
fn boundary_constraints(
        &self,
        _rap_challenges: &[FieldElement<Self::Field>],
    ) -> BoundaryConstraints<Self::Field> {
        let a0 = BoundaryConstraint::new_simple_main(0, self.pub_inputs.a0.clone());
        let a1 = BoundaryConstraint::new_simple_main(1, self.pub_inputs.a1.clone());

        BoundaryConstraints::from_constraints(vec![a0, a1])
    }
```
To understand how this works, we will look at
```rust
let a1 = BoundaryConstraint::new_simple_main(1, self.pub_inputs.a1.clone());
```
This creates a new boundary constraint that is applied to the second row (row with index 1) and takes the value `self.pub_inputs.a1`. With all this information, we can run the prover and generate the proof.

Once we have the proof, we can check it using the verifier:
```rust
let result = Verifier::<FibonacciAIR<Stark252PrimeField>>::verify(
        &proof,
        &pub_inputs,
        &proof_options,
        StoneProverTranscript::new(&[]),
    );
```

## To test compatibility with stone prover

Fetch the submodule with the Stone fork compatibility demo with:

```git submodule update --init --recursive```

You can then cd to the downloaded Stone Prover, and follow the README instructions to make a proof with Platinum and verify it with Stone

```cd ../stone-demo```

## To be added

-  Winterfell api compatibility
-  Add more parallelization
-  Optimizations
  - Skip layers
  - Stop FRI
  - Others
-  Optimized backend for mini goldilocks
-  Pick hash configuration with ProofOptions
-  Support FFTx for CUDA
-  Tracing tools
-  Virtual columns

## Requirements

- Cargo 1.69+

## 📚 References

The following links, repos and projects have been important in the development of this library and we want to thank and acknowledge them. 

- [Starkware](https://starkware.co/)
- [Winterfell](https://github.com/facebook/winterfell)
- [Anatomy of a Stark](https://aszepieniec.github.io/stark-anatomy/overview)
- [Giza](https://github.com/maxgillett/giza)
- [Ministark](https://github.com/andrewmilson/ministark)
- [Sandstorm](https://github.com/andrewmilson/sandstorm)
- [STARK-101](https://starkware.co/stark-101/)
- [Risc0](https://github.com/risc0/risc0)
- [Neptune](https://github.com/Neptune-Crypto)
- [Summary on FRI low degree test](https://eprint.iacr.org/2022/1216)
- [STARKs paper](https://eprint.iacr.org/2018/046)
- [DEEP FRI](https://eprint.iacr.org/2019/336)
- [BrainSTARK](https://aszepieniec.github.io/stark-brainfuck/)
- [Plonky2](https://github.com/mir-protocol/plonky2)
- [Aztec](https://github.com/AztecProtocol)
- [Arkworks](https://github.com/arkworks-rs)
- [Thank goodness it's FRIday](https://vitalik.ca/general/2017/11/22/starks_part_2.html)
- [Diving DEEP FRI](https://blog.lambdaclass.com/diving-deep-fri/)
- [Periodic constraints](https://blog.lambdaclass.com/periodic-constraints-and-recursion-in-zk-starks/)
- [Chiplets Miden VM](https://wiki.polygon.technology/docs/miden/design/chiplets/main/)
- [Valida](https://github.com/valida-xyz/valida/tree/main)
- [Solidity Verifier](https://github.com/starkware-libs/starkex-contracts/tree/master/evm-verifier/solidity/contracts/cpu)
- [CAIRO verifier](https://github.com/starkware-libs/cairo-lang/tree/master/src/starkware/cairo/stark_verifier)
- [EthSTARK](https://github.com/starkware-libs/ethSTARK/tree/master)
- [CAIRO whitepaper](https://eprint.iacr.org/2021/1063.pdf)
- [Gnark](https://github.com/Consensys/gnark)

## 🌞 Related Projects

- [CAIRO VM - Rust](https://github.com/lambdaclass/cairo-vm)
- [CAIRO VM - Go](https://github.com/lambdaclass/cairo_vm.go)
- [Lambdaworks](https://github.com/lambdaclass/lambdaworks)
- [CAIRO native](https://github.com/lambdaclass/cairo_native/)
- [StarkNet in Rust](https://github.com/lambdaclass/starknet_in_rust)
- [StarkNet Stack](https://github.com/lambdaclass/starknet_stack)
