# PLONK Circuit DSL

A high-level, type-safe API for building PLONK circuits. The DSL wraps the low-level [constraint system](../constraint_system/) with typed variables and composable gadgets, making circuit development more intuitive and less error-prone.

## Disclaimer

This DSL is still in development and may contain bugs. It is not intended to be used in production yet.

## Overview

[PLONK](https://eprint.iacr.org/2019/953) (Permutations over Lagrange-bases for Oecumenical Noninteractive arguments of Knowledge) is a universal and updatable zk-SNARK. This DSL provides a developer-friendly interface for constructing PLONK circuits without dealing with the underlying constraint system directly.

The DSL builds on top of several Lambdaworks components:
- [Finite Fields](../../../../math/src/field/README.md) - Field arithmetic
- [Polynomials](../../../../math/src/polynomial/README.md) - Polynomial operations
- [FFT](../../../../math/src/fft/README.md) - Fast polynomial evaluation/interpolation
- [KZG Commitments](../../../../crypto/src/commitments/README.md) - Polynomial commitment scheme

## Design Principles

1. **Type Safety**: Leverage Rust's type system to catch errors at compile time
2. **Composability**: Gadgets compose cleanly without manual constraint wiring
3. **Readability**: Circuit code should read like the computation it proves
4. **Performance**: Zero-cost abstractions where possible
5. **Debugging**: Named variables and clear error messages

## Quick Start

Build a simple circuit that proves knowledge of a secret multiplier:

```rust
use lambdaworks_plonk::dsl::CircuitBuilder;
use lambdaworks_plonk::prover::Prover;
use lambdaworks_plonk::setup::setup;
use lambdaworks_plonk::test_utils::utils::{
    test_srs, SecureRandomFieldGenerator, KZG, ORDER_R_MINUS_1_ROOT_UNITY,
};
use lambdaworks_plonk::verifier::Verifier;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::{
    FrElement, FrField,
};

// 1. Define the circuit
let mut builder = CircuitBuilder::<FrField>::new();

// Public inputs (known to verifier)
let x = builder.public_input("x");
let y = builder.public_input("y");

// Private input (known only to prover)
let secret = builder.private_input("secret");

// Circuit logic: x * secret == y
let product = builder.mul(&x, &secret);
builder.assert_eq(&product, &y);

// 2. Build common preprocessed input
let cpi = builder.build_cpi(&ORDER_R_MINUS_1_ROOT_UNITY)?;

// 3. Create witness with concrete values
let inputs = [
    ("x", FrElement::from(4u64)),
    ("y", FrElement::from(12u64)),
    ("secret", FrElement::from(3u64)),  // 4 * 3 = 12
];

let witness = builder.build_witness(&inputs)?;

// 4. Extract public inputs
let public_inputs = builder.extract_public_inputs(&inputs)?;

// 5. Setup and prove
let srs = test_srs(cpi.n);
let kzg = KZG::new(srs);
let vk = setup(&cpi, &kzg);

let prover = Prover::new(kzg.clone(), SecureRandomFieldGenerator);
let proof = prover.prove(&witness, &public_inputs, &cpi, &vk)?;

// 6. Verify
let verifier = Verifier::new(kzg);
verifier.verify_with_result(&proof, &public_inputs, &cpi, &vk)?;
```

Notes:
- For production, load an SRS from a trusted ceremony (e.g., via `SRSManager::load`) and
  check size compatibility (`SRSManager::check_size`) instead of using `test_srs`.
- Prefer `verify_with_result` for full validation (it includes subgroup checks).
  `verify` is kept for backward compatibility and skips subgroup validation.

## Production Checklist

- SRS: Load from a trusted ceremony (`SRSManager::load`) and check size (`SRSManager::check_size`).
- RNG: Use `SecureRandomFieldGenerator` for blinding; avoid test RNGs in production.
- Verification: Use `Verifier::verify_with_result` to include subgroup checks.
- Gadgets: Provide real Poseidon parameters; `DefaultPoseidonParams` is a placeholder.
- Comparisons: Range-check inputs before `LessThan`, `GreaterThan`, etc.

### Production SRS Setup (example)

```rust
use lambdaworks_plonk::srs::SRSManager;
use lambdaworks_plonk::prover::Prover;
use lambdaworks_plonk::test_utils::utils::{SecureRandomFieldGenerator, KZG};
use lambdaworks_plonk::verifier::Verifier;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::{
    curve::BLS12381Curve,
    twist::BLS12381TwistCurve,
};
use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;

type G1Point = <BLS12381Curve as IsEllipticCurve>::PointRepresentation;
type G2Point = <BLS12381TwistCurve as IsEllipticCurve>::PointRepresentation;

// Load SRS and ensure it is large enough for the circuit size `cpi.n`
let srs = SRSManager::load::<G1Point, G2Point, _>("ceremony.srs")?;
SRSManager::check_size(&srs, cpi.n)?;

// Assumes `cpi`, `witness`, `public_inputs`, and `vk` are already built
let kzg = KZG::new(srs);
let prover = Prover::new(kzg.clone(), SecureRandomFieldGenerator);
let proof = prover.prove(&witness, &public_inputs, &cpi, &vk)?;

let verifier = Verifier::new(kzg);
verifier.verify_with_result(&proof, &public_inputs, &cpi, &vk)?;
```

## Background: PLONK Constraints

PLONK uses arithmetic gates of the form:

$$q_L \cdot a + q_R \cdot b + q_O \cdot c + q_M \cdot (a \cdot b) + q_C = 0$$

Where:
- $a$, $b$, $c$ are wire values (left, right, output)
- $q_L$, $q_R$, $q_O$ are linear selector coefficients
- $q_M$ is the multiplication selector
- $q_C$ is the constant term

The DSL abstracts this away. When you write `builder.mul(&x, &y)`, it creates a gate with $q_M = 1$ and appropriate wiring. When you write `builder.add(&x, &y)`, it sets $q_L = q_R = 1$ and $q_O = -1$.

## CircuitBuilder API

### Creating Variables

```rust
// Public inputs - visible to both prover and verifier
let x = builder.public_input("x");

// Private inputs - known only to prover (the witness)
let secret = builder.private_input("secret");

// Anonymous intermediate variables
let temp = builder.new_variable();

// Constants
let five = builder.constant(FieldElement::from(5u64));
```

### Arithmetic Operations

| Operation | Code | Constraint |
|-----------|------|------------|
| Addition | `builder.add(&a, &b)` | $a + b = c$ |
| Subtraction | `builder.sub(&a, &b)` | $a - b = c$ |
| Multiplication | `builder.mul(&a, &b)` | $a \cdot b = c$ |
| Division | `builder.div(&a, &b)` | $a = b \cdot c$ |
| Inverse | `builder.inv(&a)` | $a \cdot c = 1$ |
| Scale by constant | `builder.mul_constant(&a, k)` | $k \cdot a = c$ |
| Add constant | `builder.add_constant(&a, k)` | $a + k = c$ |

### Boolean Operations

Boolean variables are constrained to be 0 or 1 via $b \cdot (b - 1) = 0$:

```rust
// Assert a variable is boolean and get typed BoolVar
let flag = builder.assert_bool(&x);

// Logical operations (inputs must be BoolVar)
let not_flag = builder.not(&flag);           // 1 - flag
let and_result = builder.and(&a, &b);        // a * b
let or_result = builder.or(&a, &b);          // a + b - a*b
let xor_result = builder.xor(&a, &b);        // a + b - 2*a*b
```

Note: `assert_bool` returns the same variable, typed as `BoolVar`. Boolean ops assume
their inputs have already been constrained to {0, 1}.

### Conditional Selection

```rust
// result = condition ? if_true : if_false
// Implemented as: result = cond * (if_true - if_false) + if_false
let result = builder.select(&condition, &if_true, &if_false);
```

### Assertions

```rust
builder.assert_eq(&a, &b);                    // a == b
builder.assert_eq_constant(&a, FieldElement::from(42u64)); // a == 42
builder.assert_zero(&a);                      // a == 0
```

## Typed Variables

The DSL uses phantom types to distinguish variable kinds at compile time:

| Type | Description | Constraint |
|------|-------------|------------|
| `FieldVar` | Arbitrary field element | None |
| `BoolVar` | Boolean (0 or 1) | $b(b-1) = 0$ |
| `U8Var` | Marker for 8-bit value (no implicit constraint) | Use `RangeCheck::<8>` |
| `U32Var` | Marker for 32-bit value (no implicit constraint) | Use `RangeCheck::<32>` |
| `U64Var` | Marker for 64-bit value (no implicit constraint) | Use `RangeCheck::<64>` |

Currently `CircuitBuilder` produces `FieldVar` and `BoolVar`. Integer marker
types are intended for ergonomics; enforce bounds explicitly with range checks.

The `AsFieldVar` trait enables polymorphic operations:

```rust
// Works with FieldVar, BoolVar, or any integer type
let sum = builder.add(&field_var, &bool_var);
```

## Gadgets

Gadgets are reusable circuit components that encapsulate common patterns. Each gadget implements the `Gadget` trait:

```rust
pub trait Gadget<F: IsField> {
    type Input;
    type Output;

    fn synthesize(
        builder: &mut CircuitBuilder<F>,
        input: Self::Input,
    ) -> Result<Self::Output, GadgetError>;

    fn constraint_count() -> usize;
    fn name() -> &'static str;
}
```

### Available Gadgets

| Category | Gadget | Description | Constraints |
|----------|--------|-------------|-------------|
| **Arithmetic** | `RangeCheck<BITS>` | Assert $0 \leq x < 2^{BITS}$ | $2 \cdot BITS + 1$ |
| | `ToBits<BITS>` | Decompose to bits (LSB first) | $2 \cdot BITS + 1$ |
| | `FromBits` | Reconstruct from bits | $O(n)$ |
| **Comparison** | `IsZero` | Check if $x = 0$ | 3 |
| | `IsEqual` | Check if $a = b$ | 4 |
| | `LessThan<BITS>` | Check if $a < b$ (bounded) | $2(BITS+1) + 2$ |
| | `LessThanOrEqual<BITS>` | Check if $a \leq b$ | $2(BITS+1) + 3$ |
| | `GreaterThan<BITS>` | Check if $a > b$ | $2(BITS+1) + 2$ |
| | `GreaterThanOrEqual<BITS>` | Check if $a \geq b$ | $2(BITS+1) + 3$ |
| **Hash** | `PoseidonHash` | Poseidon sponge hash | $O(r \cdot width^2)$ |
| | `PoseidonTwoToOne` | Binary tree hashing | $O(width^2)$ |
| **Merkle** | `MerkleProofVerifier<DEPTH>` | Compute Merkle root | $O(DEPTH \cdot hash)$ |
| | `MerkleProofChecker<DEPTH>` | Verify Merkle proof | $O(DEPTH \cdot hash)$ |

Note: Comparison gadgets assume inputs are already range-checked to the specified bit width.

### Gadget Usage Examples

**Range Check:**
```rust
use lambdaworks_plonk::dsl::gadgets::arithmetic::RangeCheck;

// Check that x is a valid u8 (0 <= x < 256)
let output = RangeCheck::<8>::synthesize(&mut builder, x.into())?;
let bits: Vec<BoolVar> = output.bits;  // Bit decomposition (LSB first)
```

**Comparison:**
```rust
use lambdaworks_plonk::dsl::gadgets::comparison::{IsZero, LessThan};

let is_zero: BoolVar = IsZero::synthesize(&mut builder, x)?;

// Assumes a, b are in range [0, 2^8)
let a_lt_b: BoolVar = LessThan::<8>::synthesize(&mut builder, (a, b))?;
```

**Poseidon Hash:**
```rust
use lambdaworks_plonk::dsl::gadgets::poseidon::{PoseidonHash, DefaultPoseidonParams};

let hash = PoseidonHash::<DefaultPoseidonParams>::synthesize(&mut builder, inputs)?;
```
Note: `DefaultPoseidonParams` is a simplified placeholder; supply real parameters for
production deployments.

**Merkle Proof:**
```rust
use lambdaworks_plonk::dsl::gadgets::merkle::{MerkleProofVerifier, MerkleProofInput};
use lambdaworks_plonk::dsl::gadgets::poseidon::DefaultPoseidonParams;

const DEPTH: usize = 32;

let input = MerkleProofInput {
    leaf,
    path: sibling_hashes,      // Vec<FieldVar>
    path_indices: directions,  // Vec<BoolVar> - 0=left, 1=right
};

let computed_root = MerkleProofVerifier::<DEPTH, DefaultPoseidonParams>::synthesize(
    &mut builder,
    input
)?;
builder.assert_eq(&computed_root, &expected_root);
```

## Examples

### Proving Knowledge of a Preimage

```rust
use lambdaworks_plonk::dsl::gadgets::poseidon::{DefaultPoseidonParams, PoseidonHash};

let mut builder = CircuitBuilder::<FrField>::new();

let public_hash = builder.public_input("hash");
let secret = builder.private_input("secret");

// Prove: hash(secret) == public_hash
let computed = PoseidonHash::<DefaultPoseidonParams>::synthesize(&mut builder, vec![secret])?;
builder.assert_eq(&computed, &public_hash);
```

### Range-Bounded Arithmetic

```rust
let mut builder = CircuitBuilder::<FrField>::new();

let a = builder.public_input("a");
let b = builder.public_input("b");

// Ensure inputs are valid u16 values
RangeCheck::<16>::synthesize(&mut builder, a.into())?;
RangeCheck::<16>::synthesize(&mut builder, b.into())?;

// Safe to compute a < b (both bounded)
let less_than = LessThan::<16>::synthesize(&mut builder, (a, b))?;
```

### Conditional Logic

```rust
let mut builder = CircuitBuilder::<FrField>::new();

let condition = builder.public_input("condition");
let a = builder.private_input("a");
let b = builder.private_input("b");
let result = builder.public_input("result");

let cond_bool = builder.assert_bool(&condition);

// result = condition ? a : b
let selected = builder.select(&cond_bool, &a, &b);
builder.assert_eq(&selected, &result);
```

## Creating Custom Gadgets

Implement the `Gadget` trait to create reusable components:

```rust
use lambdaworks_plonk::dsl::gadgets::{Gadget, GadgetError};
use lambdaworks_plonk::dsl::{CircuitBuilder, FieldVar};

/// Computes x^3
pub struct Cube;

impl<F: IsField> Gadget<F> for Cube {
    type Input = FieldVar;
    type Output = FieldVar;

    fn synthesize(
        builder: &mut CircuitBuilder<F>,
        x: Self::Input,
    ) -> Result<Self::Output, GadgetError> {
        let x_squared = builder.mul(&x, &x);
        let x_cubed = builder.mul(&x_squared, &x);
        Ok(x_cubed)
    }

    fn constraint_count() -> usize {
        2  // Two multiplication gates
    }

    fn name() -> &'static str {
        "Cube"
    }
}
```

## Module Structure

| File | Description |
|------|-------------|
| `mod.rs` | Module exports and public API |
| `builder.rs` | `CircuitBuilder` - high-level circuit construction |
| `types.rs` | Typed variables (`FieldVar`, `BoolVar`, etc.) |
| `gadgets/mod.rs` | `Gadget` trait and `GadgetError` |
| `gadgets/arithmetic.rs` | `RangeCheck`, `ToBits`, `FromBits` |
| `gadgets/comparison.rs` | `IsZero`, `IsEqual`, `LessThan`, etc. |
| `gadgets/poseidon.rs` | Poseidon hash gadget |
| `gadgets/merkle.rs` | Merkle proof verification gadgets |
| `integration_tests.rs` | End-to-end prove/verify tests |
| `gadgets/correctness_tests.rs` | Gadget constraint correctness tests |

## Running Tests

```bash
# Run all DSL tests
cargo test -p lambdaworks-plonk dsl

# Run integration tests (full prove/verify cycles)
cargo test -p lambdaworks-plonk integration_tests

# Run gadget correctness tests
cargo test -p lambdaworks-plonk correctness_tests
```

## Benchmarks

Run benchmarks for various circuit sizes (up to $2^{15}$ constraints):

```bash
cargo bench -p lambdaworks-plonk
```

Prover complexity is $O(n \log n)$ due to FFT operations. Verifier complexity is $O(1)$ - constant regardless of circuit size (PLONK's key property).

## References

- [PLONK Paper](https://eprint.iacr.org/2019/953) - Gabizon, Williamson, Ciobotaru (2019)
- [Understanding PLONK](https://vitalik.ca/general/2019/09/22/plonk.html) - Vitalik Buterin
- [From AIRs to RAPs](https://hackmd.io/@aztec-network/plonk-arithmetiization-air) - Aztec Network
- [PLONK by Hand](https://research.metastate.dev/plonk-by-hand-part-1/) - Metastate Research
- [KZG Polynomial Commitments](https://dankradfeist.de/ethereum/2020/06/16/kate-polynomial-commitments.html) - Dankrad Feist
