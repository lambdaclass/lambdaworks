# STARK Prover (stark-platinum-prover)

The `stark-platinum-prover` crate implements the STARK (Scalable Transparent ARgument of Knowledge) proof system. STARKs provide transparent proofs (no trusted setup) with post-quantum security.

## Installation

```toml
[dependencies]
stark-platinum-prover = "0.13.0"
lambdaworks-math = "0.13.0"
```

For parallel proving:

```toml
[dependencies]
stark-platinum-prover = { version = "0.13.0", features = ["parallel"] }
```

## Overview

STARK proofs work by:

1. Expressing computation as an **execution trace** (matrix of field elements)
2. Defining **constraints** that valid traces must satisfy
3. Using **FRI** (Fast Reed-Solomon IOP) for polynomial commitments
4. Applying **Fiat-Shamir** to make the proof non-interactive

## Core Components

| Component | Description |
|-----------|-------------|
| `AIR` | Algebraic Intermediate Representation trait |
| `Prover` | Generates STARK proofs |
| `Verifier` | Verifies STARK proofs |
| `FRI` | Fast Reed-Solomon commitment scheme |
| `ProofOptions` | Configuration for security and performance |

## Defining an AIR

To prove a computation, implement the `AIR` trait:

```rust
use stark_platinum_prover::traits::AIR;
use stark_platinum_prover::constraints::boundary::BoundaryConstraints;
use stark_platinum_prover::constraints::transition::TransitionConstraint;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;

type F = Stark252PrimeField;
type FE = FieldElement<F>;

#[derive(Clone)]
pub struct MyAIR {
    pub trace_length: usize,
    pub pub_inputs: MyPublicInputs,
}

impl AIR for MyAIR {
    type Field = F;
    type FieldExtension = F;  // No extension needed
    type PublicInputs = MyPublicInputs;

    fn new(
        trace_length: usize,
        pub_inputs: &Self::PublicInputs,
        proof_options: &ProofOptions,
    ) -> Self {
        Self {
            trace_length,
            pub_inputs: pub_inputs.clone(),
        }
    }

    fn build_auxiliary_trace(
        &self,
        _trace: &mut TraceTable<Self::Field>,
        _challenges: &[FieldElement<Self::FieldExtension>],
    ) {
        // Optional: build auxiliary columns for RAP
    }

    fn compute_transition(
        &self,
        frame: &Frame<Self::Field, Self::FieldExtension>,
    ) -> Vec<FieldElement<Self::FieldExtension>> {
        // Define transition constraints
        // Return vector of constraint evaluations (should be zero for valid traces)
        let current = frame.get_evaluation_step(0);
        let next = frame.get_evaluation_step(1);

        // Example: next[0] = current[0] + current[1]
        vec![next.get_main_evaluation_element(0, 0)
            - current.get_main_evaluation_element(0, 0)
            - current.get_main_evaluation_element(0, 1)]
    }

    fn boundary_constraints(
        &self,
        _rap_challenges: &[FieldElement<Self::FieldExtension>],
    ) -> BoundaryConstraints<Self::FieldExtension> {
        // Define boundary constraints (values at specific steps)
        let mut constraints = BoundaryConstraints::new();

        // First row constraints
        constraints.add_constraint(0, 0, self.pub_inputs.a0.clone());  // col 0, row 0
        constraints.add_constraint(0, 1, self.pub_inputs.a1.clone());  // col 1, row 0

        constraints
    }

    fn context(&self) -> &AirContext {
        // Return AIR metadata
        &self.context
    }

    fn trace_length(&self) -> usize {
        self.trace_length
    }

    fn pub_inputs(&self) -> &Self::PublicInputs {
        &self.pub_inputs
    }
}
```

## Example: Fibonacci STARK

Here is a complete example proving the Fibonacci sequence:

```rust
use stark_platinum_prover::proof::options::ProofOptions;
use stark_platinum_prover::prover::{IsStarkProver, Prover};
use stark_platinum_prover::verifier::{IsStarkVerifier, Verifier};
use stark_platinum_prover::examples::simple_fibonacci::{
    FibonacciAIR, FibonacciPublicInputs, fibonacci_trace
};
use stark_platinum_prover::transcript::StoneProverTranscript;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;

type F = Stark252PrimeField;
type FE = FieldElement<F>;

fn main() {
    // 1. Generate execution trace
    let initial_values = [FE::one(), FE::one()];
    let trace_length = 8;
    let mut trace = fibonacci_trace(initial_values, trace_length);

    // 2. Set proof options
    let proof_options = ProofOptions::default_test_options();

    // 3. Define public inputs
    let pub_inputs = FibonacciPublicInputs {
        a0: FE::one(),
        a1: FE::one(),
    };

    // 4. Create transcript for Fiat-Shamir
    let transcript = StoneProverTranscript::new(&[]);

    // 5. Generate proof
    let proof = Prover::<FibonacciAIR<F>>::prove(
        &mut trace,
        &pub_inputs,
        &proof_options,
        transcript.clone(),
    ).expect("proving failed");

    println!("Proof generated successfully!");
    println!("Proof size: {} bytes", proof.serialize().len());

    // 6. Verify proof
    let valid = Verifier::<FibonacciAIR<F>>::verify(
        &proof,
        &pub_inputs,
        &proof_options,
        transcript,
    );

    assert!(valid);
    println!("Proof verified successfully!");
}
```

## Proof Options

Configure security and performance parameters:

```rust
use stark_platinum_prover::proof::options::ProofOptions;

// Default test options (low security, fast)
let test_options = ProofOptions::default_test_options();

// Custom options
let options = ProofOptions::new(
    32,    // security_bits: target security level
    4,     // blowup_factor: domain extension (higher = more secure)
    8,     // fri_number_of_queries: FRI query count
    32,    // coset_offset: offset for LDE domain
);
```

| Parameter | Description | Impact |
|-----------|-------------|--------|
| `security_bits` | Target security level | Higher = more secure |
| `blowup_factor` | LDE domain multiplier | Higher = larger proofs, more secure |
| `fri_number_of_queries` | FRI verification queries | Higher = more secure, larger proofs |
| `coset_offset` | Coset shift for evaluation | Affects soundness |

## Execution Trace

The trace is a matrix where each column is a register and each row is a state:

```rust
use stark_platinum_prover::trace::TraceTable;

// Create trace with 2 columns
let mut trace = TraceTable::<F>::new_from_cols(&[
    vec![FE::one(), FE::one(), FE::from(2), FE::from(3), FE::from(5)],
    vec![FE::one(), FE::from(2), FE::from(3), FE::from(5), FE::from(8)],
]);

// Access elements
let value = trace.get(row_index, col_index);
```

## Constraints

### Transition Constraints

Transition constraints relate consecutive rows:

```rust
fn compute_transition(&self, frame: &Frame<F, F>) -> Vec<FE> {
    let current = frame.get_evaluation_step(0);
    let next = frame.get_evaluation_step(1);

    // Constraint: next[0] = current[0] + current[1]
    // Returns: next[0] - current[0] - current[1] (should be 0)
    vec![
        next.get_main_evaluation_element(0, 0)
            - current.get_main_evaluation_element(0, 0)
            - current.get_main_evaluation_element(0, 1)
    ]
}
```

### Boundary Constraints

Boundary constraints fix values at specific positions:

```rust
fn boundary_constraints(&self, _challenges: &[FE]) -> BoundaryConstraints<F> {
    let mut bc = BoundaryConstraints::new();

    // At row 0, column 0, value must be initial_a
    bc.add_constraint(0, 0, self.pub_inputs.initial_a.clone());

    // At row 0, column 1, value must be initial_b
    bc.add_constraint(0, 1, self.pub_inputs.initial_b.clone());

    bc
}
```

## Proof Structure

```rust
use stark_platinum_prover::proof::stark::StarkProof;

// Serialize proof
let bytes = proof.serialize();

// Deserialize proof
let restored = StarkProof::deserialize(&bytes).expect("deserialization");

// Proof components
// - lde_trace_merkle_root: commitment to trace evaluations
// - composition_poly_root: commitment to composition polynomial
// - fri_layers: FRI commitment layers
// - query_list: query responses with Merkle proofs
// - deep_poly_openings: DEEP method evaluations
```

## Built-in Examples

The crate includes several example AIRs:

```rust
use stark_platinum_prover::examples::{
    simple_fibonacci::{FibonacciAIR, fibonacci_trace},
    quadratic_air::QuadraticAIR,
    fibonacci_rap::FibonacciRAP,
    bit_flags::BitFlagsAIR,
};
```

### Quadratic AIR

Proves knowledge of a square root:

```rust
use stark_platinum_prover::examples::quadratic_air::{QuadraticAIR, QuadraticPublicInputs};

// Prove: I know x such that x^2 = 25
let pub_inputs = QuadraticPublicInputs { result: FE::from(25) };
```

### Fibonacci RAP

Fibonacci with Randomized AIR with Preprocessing (auxiliary columns):

```rust
use stark_platinum_prover::examples::fibonacci_rap::FibonacciRAP;
```

## Transcript

The Fiat-Shamir transcript ensures non-interactivity:

```rust
use stark_platinum_prover::transcript::StoneProverTranscript;

// Initialize with public input
let mut transcript = StoneProverTranscript::new(&public_input_bytes);

// Prover appends commitments
transcript.append_bytes(&commitment);

// Sample challenges
let alpha = transcript.sample_field_element();
let indices = transcript.sample_u64(max_value);
```

## FRI Configuration

FRI parameters affect proof size and security:

```rust
// FRI is configured through ProofOptions
// - blowup_factor: determines Reed-Solomon code rate
// - fri_number_of_queries: number of verification queries

// Higher blowup = smaller relative degree, more secure
// More queries = lower soundness error, larger proofs
```

## Performance Tips

1. **Enable parallelism**: Use the `parallel` feature for multi-threaded proving.

2. **Batch similar proofs**: If proving many similar statements, consider batching.

3. **Optimize trace**: Minimize trace width (columns) and height (rows) while maintaining constraint simplicity.

4. **Use appropriate field**: Smaller fields (BabyBear, Mersenne31) are faster but have lower security margin.

## WebAssembly Support

```toml
[dependencies]
stark-platinum-prover = { version = "0.13.0", features = ["wasm"] }
```

The STARK prover can be compiled to WebAssembly for browser verification.

## Further Reading

1. [STARK Anatomy](https://aszepieniec.github.io/stark-anatomy/) - In-depth STARK tutorial
2. [STARKs paper](https://eprint.iacr.org/2018/046) - Original STARK construction
3. [DEEP FRI](https://eprint.iacr.org/2019/336) - FRI with DEEP method
4. [FRI soundness](https://eprint.iacr.org/2022/1216) - FRI security analysis
