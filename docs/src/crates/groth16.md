# Groth16 Prover (lambdaworks-groth16)

The `lambdaworks-groth16` crate implements the Groth16 zk-SNARK, the most succinct proof system producing the smallest proofs (~200 bytes). Groth16 requires a per-circuit trusted setup.

## Installation

```toml
[dependencies]
lambdaworks-groth16 = "0.13.0"
lambdaworks-math = "0.13.0"
```

For Circom integration:

```toml
[dependencies]
lambdaworks-circom-adapter = "0.13.0"
```

## Overview

Groth16 uses:

1. **R1CS** (Rank-1 Constraint System): Circuit representation
2. **QAP** (Quadratic Arithmetic Program): Polynomial encoding
3. **Pairing-based verification**: Constant-time verification using elliptic curve pairings
4. **Per-circuit setup**: Separate trusted setup for each circuit

## Core Components

| Component | Description |
|-----------|-------------|
| `R1CS` | Rank-1 Constraint System representation |
| `QAP` | Quadratic Arithmetic Program |
| `Prover` | Generates Groth16 proofs |
| `verify` | Verifies Groth16 proofs |
| `setup` | Generates proving and verification keys |

## R1CS Format

An R1CS constraint has the form:

$$(A \cdot s) \circ (B \cdot s) = C \cdot s$$

where $s$ is the witness vector, $A$, $B$, $C$ are constraint matrices, and $\circ$ is element-wise multiplication.

### Defining R1CS Manually

```rust
use lambdaworks_groth16::r1cs::{R1CS, Constraint};
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bn_254::default_types::FrElement;

type FE = FrElement;

// Witness: [1, public_input, private_witness...]
// Index 0 is always 1 (for constants)

// Example: prove x * x = y
// Constraint: (x) * (x) = (y)
// A: [0, 0, 1]  (coefficient of x in left side)
// B: [0, 0, 1]  (coefficient of x in right side)
// C: [0, 1, 0]  (coefficient of y in output)

let constraint = Constraint {
    a: vec![(2, FE::one())],  // x is at index 2
    b: vec![(2, FE::one())],  // x is at index 2
    c: vec![(1, FE::one())],  // y is at index 1 (public)
};

let r1cs = R1CS {
    num_variables: 3,
    num_public_inputs: 1,
    constraints: vec![constraint],
};
```

### Loading from Circom

```rust
use lambdaworks_circom_adapter::CircomAdapter;

// Load R1CS and witness from Circom output
let adapter = CircomAdapter::from_files(
    "circuit.r1cs",
    "witness.wtns",
).expect("loading circuit");

let (r1cs, witness) = adapter.to_lambdaworks();
```

## Trusted Setup

Generate proving and verification keys:

```rust
use lambdaworks_groth16::setup::{setup, ProvingKey, VerificationKey};

// Generate keys (uses random toxic waste internally)
let (proving_key, verification_key) = setup(&r1cs)
    .expect("setup");

// In production, use a multi-party computation ceremony
// to generate the setup without any party knowing the full secret
```

## Proof Generation

```rust
use lambdaworks_groth16::{Prover, Proof};

// Witness: [1, public_input_1, ..., private_witness_1, ...]
let witness = vec![
    FE::one(),           // Constant 1 (always at index 0)
    FE::from(9),         // Public input: y = 9
    FE::from(3),         // Private witness: x = 3
];

// Generate proof
let proof = Prover::prove(&proving_key, &r1cs, &witness)
    .expect("proving");

println!("Proof generated: A, B, C group elements");
```

## Verification

```rust
use lambdaworks_groth16::verify;

// Extract public inputs (everything except constant 1 and private witnesses)
let public_inputs = vec![FE::from(9)];

// Verify
let is_valid = verify(&verification_key, &proof, &public_inputs);

assert!(is_valid);
println!("Proof verified!");
```

## Complete Example

Prove knowledge of a square root:

```rust
use lambdaworks_groth16::{R1CS, Constraint, Prover, verify, setup};
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bn_254::default_types::FrElement;

type FE = FrElement;

fn main() {
    // Circuit: prove knowledge of x such that x * x = y
    // Witness format: [1, y, x] where y is public, x is private

    // Constraint: x * x = y
    let constraint = Constraint {
        a: vec![(2, FE::one())],  // x
        b: vec![(2, FE::one())],  // x
        c: vec![(1, FE::one())],  // y
    };

    let r1cs = R1CS {
        num_variables: 3,         // [1, y, x]
        num_public_inputs: 1,     // y is public
        constraints: vec![constraint],
    };

    // Setup
    let (pk, vk) = setup(&r1cs).expect("setup");

    // Prove: x = 5, y = 25
    let witness = vec![
        FE::one(),      // constant
        FE::from(25),   // y (public)
        FE::from(5),    // x (private)
    ];

    let proof = Prover::prove(&pk, &r1cs, &witness).expect("prove");

    // Verify
    let public_inputs = vec![FE::from(25)];
    assert!(verify(&vk, &proof, &public_inputs));

    println!("Proved: I know sqrt(25)");
}
```

## Circom Integration

### Workflow

1. Write circuit in Circom
2. Compile to R1CS and generate witness
3. Load into lambdaworks
4. Generate and verify proof

### Example: Fibonacci Circuit

**circuit.circom:**
```circom
pragma circom 2.0.0;

template Fibonacci(n) {
    signal input in[2];
    signal output out;

    signal fib[n+1];
    fib[0] <== in[0];
    fib[1] <== in[1];

    for (var i = 2; i <= n; i++) {
        fib[i] <== fib[i-1] + fib[i-2];
    }

    out <== fib[n];
}

component main = Fibonacci(10);
```

**Compile and prove:**
```bash
# Compile circuit
circom circuit.circom --r1cs --wasm

# Generate witness
node circuit_js/generate_witness.js circuit_js/circuit.wasm input.json witness.wtns
```

**Rust code:**
```rust
use lambdaworks_circom_adapter::CircomAdapter;
use lambdaworks_groth16::{setup, Prover, verify};

fn main() {
    // Load from Circom output
    let adapter = CircomAdapter::from_files(
        "circuit.r1cs",
        "witness.wtns",
    ).expect("load");

    let (r1cs, witness) = adapter.to_lambdaworks();

    // Setup and prove
    let (pk, vk) = setup(&r1cs).expect("setup");
    let proof = Prover::prove(&pk, &r1cs, &witness).expect("prove");

    // Verify
    let public_inputs = adapter.get_public_inputs();
    assert!(verify(&vk, &proof, &public_inputs));
}
```

## Arkworks Compatibility

For interoperability with Arkworks:

```rust
use lambdaworks_groth16::arkworks_adapter::ArkworksAdapter;

// Load Arkworks R1CS
let adapter = ArkworksAdapter::from_arkworks_r1cs(ark_r1cs);
let (r1cs, witness) = adapter.to_lambdaworks();
```

## Proof Structure

A Groth16 proof consists of three group elements:

```rust
pub struct Proof<G1, G2> {
    pub a: G1,   // [A]_1 - element in G1
    pub b: G2,   // [B]_2 - element in G2
    pub c: G1,   // [C]_1 - element in G1
}
```

The proof is verified by checking:

$$e(A, B) = e(\alpha, \beta) \cdot e(L, \gamma) \cdot e(C, \delta)$$

where $\alpha, \beta, \gamma, \delta$ are from the verification key and $L$ encodes public inputs.

## Supported Curves

Currently supported:
1. BN254 (Ethereum-compatible)
2. BLS12-381

## Performance

| Metric | Value |
|--------|-------|
| Proof size | ~200 bytes (2 G1 + 1 G2 points) |
| Verification | ~3 pairings |
| Prover time | O(n log n) MSM operations |

## Security Considerations

1. **Trusted Setup**: The setup phase produces "toxic waste" that must be destroyed. If compromised, fake proofs can be created. Use multi-party computation ceremonies.

2. **Circuit Soundness**: The R1CS must correctly encode your computation. A bug in the circuit can allow invalid proofs.

3. **Witness Privacy**: Private witnesses are not revealed, but ensure your circuit doesn't leak information through public inputs.

## Error Handling

```rust
use lambdaworks_groth16::Groth16Error;

match Prover::prove(&pk, &r1cs, &witness) {
    Ok(proof) => println!("Success!"),
    Err(Groth16Error::InvalidWitness) => println!("Witness doesn't satisfy constraints"),
    Err(Groth16Error::IncorrectWitnessLength) => println!("Wrong number of witness elements"),
    Err(e) => println!("Error: {:?}", e),
}
```

## Further Reading

1. [Groth16 Paper](https://eprint.iacr.org/2016/260) - Original construction
2. [Circom Documentation](https://docs.circom.io/) - Circuit language
3. [Arkworks](https://arkworks.rs/) - Rust zkSNARK library
4. [ZK Security](https://blog.trailofbits.com/2022/04/18/the-security-of-zero-knowledge-proofs/) - Security considerations
