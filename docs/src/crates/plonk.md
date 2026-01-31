# PLONK Prover (lambdaworks-plonk)

The `lambdaworks-plonk` crate implements the PLONK (Permutations over Lagrange-bases for Oecumenical Noninteractive arguments of Knowledge) proof system. PLONK provides succinct proofs with a universal trusted setup.

## Installation

```toml
[dependencies]
lambdaworks-plonk = "0.13.0"
lambdaworks-math = "0.13.0"
lambdaworks-crypto = "0.13.0"
```

## Overview

PLONK is a universal SNARK that uses:

1. **Custom gates**: Flexible constraint format supporting addition and multiplication
2. **Copy constraints**: Wire connections via permutation arguments
3. **KZG commitments**: Polynomial commitments using pairings
4. **Universal setup**: Single SRS works for any circuit up to a size limit

## Core Components

| Component | Description |
|-----------|-------------|
| `ConstraintSystem` | Circuit builder API |
| `Prover` | Generates PLONK proofs |
| `Verifier` | Verifies PLONK proofs |
| `Setup` | Key generation from SRS |

## Circuit Definition

### Constraint System API

Build circuits using the constraint system:

```rust
use lambdaworks_plonk::constraint_system::ConstraintSystem;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrElement;

type FE = FrElement;

// Create constraint system
let mut cs = ConstraintSystem::<FE>::new();

// Add witness variables
let x = cs.add_variable(FE::from(3));       // Private input
let x_sq = cs.add_variable(FE::from(9));    // x^2
let x_cubed = cs.add_variable(FE::from(27)); // x^3

// Add multiplication gate: x * x = x_sq
cs.add_mul_constraint(x, x, x_sq);

// Add multiplication gate: x * x_sq = x_cubed
cs.add_mul_constraint(x, x_sq, x_cubed);

// Add addition gate: a + b = c
let a = cs.add_variable(FE::from(5));
let b = cs.add_variable(FE::from(7));
let c = cs.add_variable(FE::from(12));
cs.add_add_constraint(a, b, c);
```

### Gate Types

PLONK uses a universal gate equation:

$$q_L \cdot a + q_R \cdot b + q_O \cdot c + q_M \cdot a \cdot b + q_C = 0$$

The constraint system provides convenient methods:

```rust
// Addition: a + b = c
// Sets q_L = 1, q_R = 1, q_O = -1, q_M = 0, q_C = 0
cs.add_add_constraint(a, b, c);

// Multiplication: a * b = c
// Sets q_L = 0, q_R = 0, q_O = -1, q_M = 1, q_C = 0
cs.add_mul_constraint(a, b, c);

// Constant: a = constant
// Sets q_L = 1, q_R = 0, q_O = 0, q_M = 0, q_C = -constant
cs.add_constant_constraint(a, constant);

// Boolean: a * (1 - a) = 0
cs.add_boolean_constraint(a);
```

### Public Inputs

Mark variables as public:

```rust
// Add public input
let public_output = cs.add_public_variable(FE::from(35));

// The prover must provide values matching public inputs
// The verifier checks against these values
```

## Proving and Verification

### Setup

Generate proving and verification keys:

```rust
use lambdaworks_plonk::setup::setup;
use lambdaworks_crypto::commitments::kzg::StructuredReferenceString;

// Load or generate SRS (from trusted setup ceremony)
let srs = StructuredReferenceString::from_file("srs.bin")
    .expect("SRS loading");

// Generate keys from circuit
let (proving_key, verification_key) = setup(&cs, &srs)
    .expect("setup");
```

### Proof Generation

```rust
use lambdaworks_plonk::prover::Prover;

// Create witness assignment
let witness = cs.get_witness();

// Generate proof
let proof = Prover::prove(
    &proving_key,
    &cs,
    &witness,
).expect("proving");

println!("Proof generated!");
```

### Verification

```rust
use lambdaworks_plonk::verifier::verify;

// Extract public inputs
let public_inputs = cs.get_public_inputs();

// Verify proof
let is_valid = verify(
    &verification_key,
    &proof,
    &public_inputs,
);

assert!(is_valid);
println!("Proof verified!");
```

## Complete Example

Here is a complete example proving knowledge of a cube root:

```rust
use lambdaworks_plonk::constraint_system::ConstraintSystem;
use lambdaworks_plonk::prover::Prover;
use lambdaworks_plonk::verifier::verify;
use lambdaworks_plonk::setup::setup;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrElement;

type FE = FrElement;

fn main() {
    // Circuit: prove knowledge of x such that x^3 = 27
    // (x = 3)

    let mut cs = ConstraintSystem::<FE>::new();

    // Private witness: x = 3
    let x = cs.add_variable(FE::from(3));

    // Intermediate: x^2 = 9
    let x_sq = cs.add_variable(FE::from(9));
    cs.add_mul_constraint(x, x, x_sq);

    // Result: x^3 = 27 (public)
    let x_cubed = cs.add_public_variable(FE::from(27));
    cs.add_mul_constraint(x, x_sq, x_cubed);

    // Setup (in production, use ceremony-generated SRS)
    let srs = generate_test_srs(cs.num_constraints());
    let (pk, vk) = setup(&cs, &srs).expect("setup");

    // Prove
    let witness = cs.get_witness();
    let proof = Prover::prove(&pk, &cs, &witness).expect("prove");

    // Verify
    let public_inputs = vec![FE::from(27)];
    assert!(verify(&vk, &proof, &public_inputs));

    println!("Proved: I know x such that x^3 = 27");
}
```

## Advanced Features

### Range Constraints

Prove a value is within a range:

```rust
// Prove 0 <= x < 2^n using binary decomposition
fn add_range_constraint(cs: &mut ConstraintSystem<FE>, x: Variable, n: usize) {
    let mut bits = Vec::new();
    let x_val = cs.get_variable(x);

    // Decompose into bits
    for i in 0..n {
        let bit_val = (x_val.representative() >> i) & 1;
        let bit = cs.add_variable(FE::from(bit_val));
        cs.add_boolean_constraint(bit);
        bits.push(bit);
    }

    // Reconstruct and check equality
    let reconstructed = cs.linear_combination(&bits, |i| FE::from(1u64 << i));
    cs.add_equality_constraint(x, reconstructed);
}
```

### Lookup Tables

For efficient range checks and other patterns, consider lookup arguments (future feature).

## Proof Structure

A PLONK proof contains:

```rust
pub struct PlonkProof<G1, G2> {
    // Wire commitments
    pub a_commitment: G1,
    pub b_commitment: G1,
    pub c_commitment: G1,

    // Permutation commitment
    pub z_commitment: G1,

    // Quotient polynomial commitments
    pub t_lo_commitment: G1,
    pub t_mid_commitment: G1,
    pub t_hi_commitment: G1,

    // Opening evaluations
    pub a_eval: FE,
    pub b_eval: FE,
    pub c_eval: FE,
    pub s_sigma1_eval: FE,
    pub s_sigma2_eval: FE,
    pub z_shifted_eval: FE,

    // Opening proofs
    pub w_commitment: G1,
    pub w_zeta_commitment: G1,
}
```

## Performance Considerations

### Constraint Count

The number of constraints directly affects:
1. Proving time (linear in constraints)
2. SRS size requirement
3. Verification time (constant, but affected by public input count)

### Optimization Tips

1. **Minimize constraints**: Reuse intermediate values
2. **Batch similar operations**: Group related computations
3. **Use custom gates**: If extending PLONK, add specialized gates
4. **Parallelize proving**: Enable parallel features when available

## Comparison with Other Provers

| Feature | PLONK | Groth16 | STARK |
|---------|-------|---------|-------|
| **Setup** | Universal | Per-circuit | None |
| **Proof size** | ~1 KB | ~200 bytes | ~100 KB |
| **Prover time** | Medium | Slow | Fast |
| **Quantum-safe** | No | No | Yes |

## Current Limitations

The lambdaworks PLONK implementation is under active development. Current status:

1. Basic gates (add, mul, constant)
2. Copy constraints
3. KZG-based commitments
4. BLS12-381 curve support

Planned features:
1. Custom gates
2. Lookup arguments
3. Recursive proof composition
4. Additional curve support

## Further Reading

1. [PLONK Paper](https://eprint.iacr.org/2019/953) - Original PLONK construction
2. [Understanding PLONK](https://vitalik.ca/general/2019/09/22/plonk.html) - Vitalik's explanation
3. [PLONK by Hand](https://research.metastate.dev/plonk-by-hand-part-1/) - Step-by-step walkthrough
4. [Custom Gates](https://zcash.github.io/halo2/concepts/arithmetization.html) - Extending PLONK
