# API Overview

This document provides a quick reference to the main types and functions in lambdaworks.

## lambdaworks-math

### Field Elements

```rust
use lambdaworks_math::field::element::FieldElement;

// Creation
FieldElement::<F>::from(value: u64)
FieldElement::<F>::from_hex_unchecked(hex: &str)
FieldElement::<F>::from_bytes_be(bytes: &[u8])
FieldElement::<F>::from_bytes_le(bytes: &[u8])
FieldElement::<F>::zero()
FieldElement::<F>::one()

// Arithmetic (implemented for &FieldElement and FieldElement)
a + b, a - b, a * b, a / b, -a

// Methods
element.square() -> FieldElement
element.pow(exp: u64) -> FieldElement
element.inv() -> Result<FieldElement, FieldError>
element.sqrt() -> Option<(FieldElement, FieldElement)>

// Conversion
element.representative() -> UnsignedInteger
element.to_bytes_be() -> Vec<u8>
element.to_bytes_le() -> Vec<u8>

// Batch operations
FieldElement::inplace_batch_inverse(elements: &mut [FieldElement])
```

### Polynomials

```rust
use lambdaworks_math::polynomial::Polynomial;

// Creation
Polynomial::new(coefficients: &[FE])
Polynomial::new_monomial(coeff: FE, degree: usize)
Polynomial::zero()

// Evaluation
poly.evaluate(point: &FE) -> FE
poly.evaluate_slice(points: &[FE]) -> Vec<FE>

// Interpolation
Polynomial::interpolate(xs: &[FE], ys: &[FE]) -> Result<Polynomial>

// Arithmetic
poly1 + poly2, poly1 - poly2, poly1 * poly2, poly1 / poly2

// Division
poly.div_with_ref(divisor: &Polynomial) -> (Polynomial, Polynomial)
poly.ruffini_division(root: &FE) -> (Polynomial, FE)

// Properties
poly.degree() -> usize
poly.leading_coefficient() -> FE
poly.is_zero() -> bool
```

### Elliptic Curves

```rust
use lambdaworks_math::elliptic_curve::traits::IsEllipticCurve;
use lambdaworks_math::cyclic_group::IsGroup;

// Get generator
Curve::generator() -> Point

// Create from affine coordinates
Curve::create_point_from_affine(x: FE, y: FE) -> Result<Point>

// Point operations
point.operate_with(&other) -> Point  // Addition
point.operate_with_self(scalar: u64) -> Point  // Scalar multiplication
point.neg() -> Point  // Negation
Point::neutral_element() -> Point  // Identity

// Coordinate conversion
point.to_affine() -> AffinePoint
affine.x() -> FE
affine.y() -> FE

// Pairings (for pairing-friendly curves)
Pairing::compute_batch(&[(g1_point, g2_point), ...]) -> Result<GT>
```

### FFT

```rust
use lambdaworks_math::fft::polynomial::{evaluate_fft, interpolate_fft, multiply_fft};
use lambdaworks_math::fft::cpu::roots_of_unity::get_powers_of_primitive_root;

// FFT operations
evaluate_fft(polynomial: &Polynomial) -> Result<Vec<FE>>
interpolate_fft(evaluations: &[FE]) -> Result<Polynomial>
multiply_fft(p1: &Polynomial, p2: &Polynomial) -> Result<Polynomial>

// Roots of unity
get_primitive_root_of_unity::<F>(order: u64) -> Result<FE>
get_powers_of_primitive_root::<F>(root_order: u64, count: usize) -> Result<Vec<FE>>
```

### MSM

```rust
use lambdaworks_math::msm::pippenger::msm;

// Multi-scalar multiplication
msm(scalars: &[UnsignedInteger], points: &[Point]) -> Result<Point>
```

## lambdaworks-crypto

### Merkle Trees

```rust
use lambdaworks_crypto::merkle_tree::merkle::MerkleTree;
use lambdaworks_crypto::merkle_tree::proof::Proof;

// Build tree
MerkleTree::<Backend>::build(values: &[T]) -> Result<MerkleTree>

// Access root
tree.root -> Hash

// Generate proof
tree.get_proof_by_pos(index: usize) -> Result<Proof>

// Verify proof
proof.verify::<Backend>(root: &Hash, index: usize, value: &T) -> bool
```

### KZG Commitments

```rust
use lambdaworks_crypto::commitments::kzg::KateZaveruchaGoldberg;
use lambdaworks_crypto::commitments::traits::IsCommitmentScheme;

// Load SRS
StructuredReferenceString::from_file(path: &str) -> Result<SRS>
StructuredReferenceString::deserialize(bytes: &[u8]) -> Result<SRS>

// Create KZG instance
KateZaveruchaGoldberg::new(srs: SRS) -> KZG

// Commit
kzg.commit(polynomial: &Polynomial) -> Commitment

// Open
kzg.open(point: &FE, evaluation: &FE, polynomial: &Polynomial) -> Proof

// Verify
kzg.verify(point: &FE, evaluation: &FE, commitment: &Commitment, proof: &Proof) -> bool

// Batch operations
kzg.open_batch(point: &FE, evaluations: &[FE], polynomials: &[Polynomial], challenge: &FE) -> Proof
kzg.verify_batch(point: &FE, evaluations: &[FE], commitments: &[Commitment], proof: &Proof, challenge: &FE) -> bool
```

### Hash Functions

```rust
// Poseidon
use lambdaworks_crypto::hash::poseidon::starknet::PoseidonCairoStark252;
let hasher = PoseidonCairoStark252::new();
let hash = hasher.hash(&[FE::from(1), FE::from(2)]);

// Pedersen
use lambdaworks_crypto::hash::pedersen::Pedersen;
let hash = Pedersen::<Curve>::hash(&a, &b);
```

## stark-platinum-prover

### Proving and Verification

```rust
use stark_platinum_prover::prover::{IsStarkProver, Prover};
use stark_platinum_prover::verifier::{IsStarkVerifier, Verifier};
use stark_platinum_prover::proof::options::ProofOptions;
use stark_platinum_prover::transcript::StoneProverTranscript;

// Configure options
ProofOptions::default_test_options() -> ProofOptions
ProofOptions::new(security_bits, blowup_factor, fri_queries, coset_offset) -> ProofOptions

// Create transcript
StoneProverTranscript::new(public_input: &[u8]) -> Transcript

// Prove
Prover::<AIR>::prove(
    trace: &mut TraceTable,
    pub_inputs: &PublicInputs,
    options: &ProofOptions,
    transcript: Transcript
) -> Result<StarkProof>

// Verify
Verifier::<AIR>::verify(
    proof: &StarkProof,
    pub_inputs: &PublicInputs,
    options: &ProofOptions,
    transcript: Transcript
) -> bool

// Serialize/deserialize proof
proof.serialize() -> Vec<u8>
StarkProof::deserialize(bytes: &[u8]) -> Result<StarkProof>
```

### Trace Table

```rust
use stark_platinum_prover::trace::TraceTable;

// Create from columns
TraceTable::new_from_cols(columns: &[Vec<FE>]) -> TraceTable

// Access elements
trace.get(row: usize, col: usize) -> FE
trace.num_rows() -> usize
trace.num_cols() -> usize
```

### AIR Trait

```rust
pub trait AIR {
    type Field: IsField;
    type FieldExtension: IsField;
    type PublicInputs;

    fn new(trace_length: usize, pub_inputs: &Self::PublicInputs, options: &ProofOptions) -> Self;
    fn compute_transition(&self, frame: &Frame) -> Vec<FE>;
    fn boundary_constraints(&self, challenges: &[FE]) -> BoundaryConstraints;
    fn trace_length(&self) -> usize;
    fn pub_inputs(&self) -> &Self::PublicInputs;
}
```

## lambdaworks-groth16

```rust
use lambdaworks_groth16::{setup, Prover, Proof, verify, R1CS, Constraint};

// Define R1CS
let r1cs = R1CS {
    num_variables: usize,
    num_public_inputs: usize,
    constraints: Vec<Constraint>,
};

// Setup
setup(r1cs: &R1CS) -> Result<(ProvingKey, VerificationKey)>

// Prove
Prover::prove(pk: &ProvingKey, r1cs: &R1CS, witness: &[FE]) -> Result<Proof>

// Verify
verify(vk: &VerificationKey, proof: &Proof, public_inputs: &[FE]) -> bool
```

## lambdaworks-plonk

```rust
use lambdaworks_plonk::constraint_system::ConstraintSystem;
use lambdaworks_plonk::setup::setup;
use lambdaworks_plonk::prover::Prover;
use lambdaworks_plonk::verifier::verify;

// Build circuit
let mut cs = ConstraintSystem::new();
let var = cs.add_variable(value: FE);
let pub_var = cs.add_public_variable(value: FE);

// Add constraints
cs.add_add_constraint(a, b, c);    // a + b = c
cs.add_mul_constraint(a, b, c);    // a * b = c
cs.add_constant_constraint(a, val); // a = val
cs.add_boolean_constraint(a);       // a * (1 - a) = 0

// Setup
setup(cs: &ConstraintSystem, srs: &SRS) -> Result<(ProvingKey, VerificationKey)>

// Prove
Prover::prove(pk: &ProvingKey, cs: &ConstraintSystem, witness: &Witness) -> Result<Proof>

// Verify
verify(vk: &VerificationKey, proof: &Proof, public_inputs: &[FE]) -> bool
```

## Common Type Aliases

```rust
// Field elements
type Felt252 = FieldElement<Stark252PrimeField>;
type FrElement = FieldElement<BLS12381FrField>;
type FqElement = FieldElement<BLS12381FqField>;

// Curves
type G1Point = ShortWeierstrassProjectivePoint<BLS12381Curve>;
type G2Point = ShortWeierstrassProjectivePoint<BLS12381TwistCurve>;

// Big integers
type U256 = UnsignedInteger<4>;
type U384 = UnsignedInteger<6>;
```

## Error Types

```rust
// Field errors
enum FieldError {
    InverseOfZero,
    InvalidValue,
}

// Curve errors
enum CurveError {
    PointNotOnCurve,
}

// Proof errors
enum ProofError {
    InvalidWitness,
    ConstraintNotSatisfied,
    InvalidProof,
}
```
