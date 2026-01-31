# Proof Systems

Zero-knowledge proof systems allow a prover to convince a verifier that a statement is true without revealing anything beyond the validity of the statement. lambdaworks implements three major proof systems: STARK, PLONK, and Groth16.

## What is a Zero-Knowledge Proof?

A zero-knowledge proof system has three properties:

1. **Completeness**: If the statement is true, an honest prover can convince the verifier.
2. **Soundness**: If the statement is false, no prover can convince the verifier (except with negligible probability).
3. **Zero-Knowledge**: The proof reveals nothing beyond the truth of the statement.

## Proof System Landscape

| Property | STARK | PLONK | Groth16 |
|----------|-------|-------|---------|
| **Setup** | Transparent | Universal | Circuit-specific |
| **Proof size** | ~100 KB | ~1 KB | ~200 bytes |
| **Prover time** | Fast | Medium | Slow |
| **Verifier time** | ~ms | ~ms | ~ms |
| **Quantum-safe** | Yes | No | No |
| **Assumption** | Hash | DLog + Pairing | DLog + Pairing |

## STARK (Scalable Transparent ARgument of Knowledge)

STARKs are proof systems based on polynomials and hash functions. They require no trusted setup and are believed to be quantum-resistant.

### Core Concepts

**AIR (Algebraic Intermediate Representation)**: The computation is expressed as a set of polynomial constraints over an execution trace. Each row of the trace represents a step of the computation.

**Execution Trace**: A matrix where each column is a register and each row is a state. Transition constraints relate consecutive rows.

**FRI Commitment**: Polynomials are committed using the FRI protocol, which proves that a function is close to a low-degree polynomial.

### STARK Architecture

```
Computation
    |
    v
Execution Trace (matrix of field elements)
    |
    v
Interpolate to Polynomials (one per column)
    |
    v
Compose with Constraints (boundary + transition)
    |
    v
FRI Commitment (Merkle tree of evaluations)
    |
    v
STARK Proof
```

### Example: Fibonacci STARK

```rust
use stark_platinum_prover::prover::{IsStarkProver, Prover};
use stark_platinum_prover::verifier::{IsStarkVerifier, Verifier};
use stark_platinum_prover::examples::simple_fibonacci::{
    FibonacciAIR, FibonacciPublicInputs, fibonacci_trace
};

// Define the computation via AIR
type F = Stark252PrimeField;

// Generate execution trace
let trace = fibonacci_trace([Felt::one(), Felt::one()], 8);

// Public inputs
let pub_inputs = FibonacciPublicInputs {
    a0: Felt::one(),
    a1: Felt::one(),
};

// Prove
let proof = Prover::<FibonacciAIR<F>>::prove(
    &mut trace,
    &pub_inputs,
    &proof_options,
    transcript,
).unwrap();

// Verify
let valid = Verifier::<FibonacciAIR<F>>::verify(
    &proof,
    &pub_inputs,
    &proof_options,
    transcript,
);
```

### When to Use STARKs

Choose STARKs when:
1. You need transparency (no trusted setup).
2. Quantum resistance is important.
3. Proof size is acceptable (typically ~100 KB).
4. The computation is naturally expressed as an execution trace.

## PLONK (Permutations over Lagrange-bases for Oecumenical Noninteractive arguments of Knowledge)

PLONK is a universal SNARK that uses a single trusted setup for all circuits up to a certain size.

### Core Concepts

**Universal Setup**: One SRS works for any circuit, unlike Groth16's per-circuit setup.

**Gates**: PLONK uses a custom gate equation:
$$q_L \cdot a + q_R \cdot b + q_O \cdot c + q_M \cdot a \cdot b + q_C = 0$$

**Copy Constraints**: Wire connections are enforced through permutation arguments.

**KZG Commitments**: Polynomials are committed using the KZG scheme.

### PLONK Architecture

```
Circuit Description
    |
    v
Witness Generation (private inputs)
    |
    v
Constraint Polynomials (gates + wiring)
    |
    v
KZG Commitments
    |
    v
Fiat-Shamir Challenges
    |
    v
PLONK Proof (O(1) group elements)
```

### Example: Simple PLONK Circuit

```rust
use lambdaworks_plonk::constraint_system::ConstraintSystem;
use lambdaworks_plonk::prover::Prover;
use lambdaworks_plonk::verifier::verify;

// Define circuit: prove knowledge of x such that x^3 + x + 5 = 35
// (x = 3)

let mut cs = ConstraintSystem::new();

// Create witnesses
let x = cs.add_witness(FE::from(3));
let x_sq = cs.add_witness(FE::from(9));
let x_cubed = cs.add_witness(FE::from(27));
let result = cs.add_witness(FE::from(35));

// Add constraints
cs.add_mul_gate(x, x, x_sq);       // x * x = x^2
cs.add_mul_gate(x, x_sq, x_cubed); // x * x^2 = x^3
cs.add_add_gate(x_cubed, x, result); // x^3 + x = result - 5

// Generate proof
let proof = Prover::prove(&cs, &witness, &srs).unwrap();

// Verify
let valid = verify(&proof, &public_inputs, &srs);
```

### When to Use PLONK

Choose PLONK when:
1. You need small proofs (~1 KB).
2. Universal setup is acceptable.
3. Circuit expressiveness matters (custom gates).
4. Multiple circuits share the same setup.

## Groth16

Groth16 is the most succinct SNARK, producing the smallest proofs. It requires a per-circuit trusted setup.

### Core Concepts

**R1CS (Rank-1 Constraint System)**: Constraints are expressed as:
$$A \cdot s \circ B \cdot s = C \cdot s$$
where $s$ is the witness vector and $\circ$ is element-wise multiplication.

**QAP (Quadratic Arithmetic Program)**: R1CS is converted to polynomial form.

**Trusted Setup**: Circuit-specific parameters are generated from secret randomness that must be destroyed.

### Groth16 Architecture

```
Circuit (R1CS format)
    |
    v
QAP Transformation
    |
    v
Trusted Setup (circuit-specific)
    |
    v
Witness Assignment
    |
    v
Groth16 Proof (3 group elements)
```

### Example: Groth16 Proof

```rust
use lambdaworks_groth16::{Prover, verify, setup, R1CS};

// Define circuit in R1CS form
let r1cs = R1CS::from_file("circuit.r1cs").unwrap();

// Generate trusted setup (in production, use MPC ceremony)
let (proving_key, verification_key) = setup(&r1cs);

// Create witness
let witness = vec![
    FE::one(),      // constant 1
    FE::from(3),    // public input
    FE::from(9),    // private witness
];

// Generate proof
let proof = Prover::prove(&proving_key, &r1cs, &witness).unwrap();

// Verify
let public_inputs = vec![FE::from(3)];
let valid = verify(&verification_key, &proof, &public_inputs);
```

### Circom Integration

lambdaworks supports Circom circuits via the circom adapter:

```rust
use lambdaworks_circom_adapter::CircomAdapter;

// Load Circom circuit
let adapter = CircomAdapter::from_files(
    "circuit.r1cs",
    "witness.wtns",
).unwrap();

// Convert to lambdaworks format
let (r1cs, witness) = adapter.to_lambdaworks();

// Use with Groth16 prover
let proof = Prover::prove(&proving_key, &r1cs, &witness).unwrap();
```

### When to Use Groth16

Choose Groth16 when:
1. Proof size is critical (~200 bytes).
2. Verification gas cost matters (Ethereum).
3. Circuit is fixed (setup is done once).
4. Trusted setup is acceptable.

## Sumcheck Protocol

The sumcheck protocol proves the sum of a multivariate polynomial over the Boolean hypercube:

$$\sum_{b \in \{0,1\}^n} p(b) = s$$

It is a building block for GKR and other advanced protocols:

```rust
use lambdaworks_sumcheck::sumcheck_prover::SumcheckProver;

// Prove that sum of p over {0,1}^n equals s
let prover = SumcheckProver::new(polynomial);
let proof = prover.prove();
```

## GKR Protocol

The GKR protocol proves the correct evaluation of layered arithmetic circuits:

```rust
use lambdaworks_gkr::GkrProver;

// Define layered circuit
let circuit = LayeredCircuit::new(layers);

// Prove correct evaluation
let proof = GkrProver::prove(&circuit, &inputs);
```

## Comparison Summary

### Proof Size

```
Groth16: ████ (200 bytes)
PLONK:   ████████████████ (1 KB)
STARK:   ████████████████████████████████████████████████ (100+ KB)
```

### Prover Time

```
STARK:   ████████ (fastest)
PLONK:   ████████████████ (medium)
Groth16: ████████████████████████████████ (slowest due to pairings)
```

### Trust Assumptions

```
STARK:   None (transparent)
PLONK:   Universal SRS (one-time)
Groth16: Per-circuit (requires ceremony)
```

## Choosing a Proof System

Use this decision tree:

1. **Need quantum resistance?** Yes -> STARK
2. **Need smallest proof?** Yes -> Groth16
3. **Need universal setup?** Yes -> PLONK
4. **Need fast prover?** Yes -> STARK
5. **Need on-chain verification?** Yes -> Groth16 or PLONK
6. **Is proof size acceptable?** (~100KB) Yes -> STARK

## Further Reading

1. [STARKs paper](https://eprint.iacr.org/2018/046) - Original STARK construction
2. [PLONK paper](https://eprint.iacr.org/2019/953) - PLONK protocol
3. [Groth16 paper](https://eprint.iacr.org/2016/260) - Groth16 SNARK
4. [Proofs, Arguments, and Zero-Knowledge](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf) - Comprehensive textbook
