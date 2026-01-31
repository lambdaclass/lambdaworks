# Commitment Schemes

Polynomial commitment schemes are cryptographic primitives that allow a prover to commit to a polynomial and later prove evaluations at specific points. They are essential components of modern proof systems.

## What is a Polynomial Commitment?

A polynomial commitment scheme consists of three operations:

1. **Commit**: Given a polynomial $p(x)$, produce a commitment $C$ that binds the prover to that specific polynomial.

2. **Open**: Given a point $z$, produce a proof $\pi$ that $p(z) = y$.

3. **Verify**: Given $(C, z, y, \pi)$, verify that the committed polynomial indeed evaluates to $y$ at $z$.

The security properties are:

1. **Binding**: A prover cannot produce valid proofs for two different polynomials from the same commitment.
2. **Hiding** (optional): The commitment reveals nothing about the polynomial.

## KZG Commitment Scheme

The Kate-Zaverucha-Goldberg (KZG) scheme uses elliptic curve pairings for polynomial commitments. It is used in PLONK, Groth16, and Ethereum's EIP-4844.

### How KZG Works

**Setup**: Generate a Structured Reference String (SRS) containing powers of a secret $\tau$:

$$\text{SRS} = \{g_1, \tau \cdot g_1, \tau^2 \cdot g_1, \ldots, \tau^{n-1} \cdot g_1\} \cup \{g_2, \tau \cdot g_2\}$$

where $g_1$ and $g_2$ are generators of the pairing groups.

**Commit**: For polynomial $p(x) = \sum_i a_i x^i$, compute:

$$C = \sum_i a_i \cdot [\tau^i]_1 = [p(\tau)]_1$$

This is a multi-scalar multiplication (MSM) with the SRS points.

**Open at point $z$**: Compute the quotient polynomial:

$$q(x) = \frac{p(x) - p(z)}{x - z}$$

The proof is $\pi = [q(\tau)]_1$.

**Verify**: Check the pairing equation:

$$e(C - [y]_1, g_2) = e(\pi, [\tau]_2 - [z]_2)$$

This works because if $p(z) = y$, then $(x - z)$ divides $(p(x) - y)$.

### Using KZG in lambdaworks

```rust
use lambdaworks_crypto::commitments::kzg::{
    KateZaveruchaGoldberg, StructuredReferenceString
};
use lambdaworks_crypto::commitments::traits::IsCommitmentScheme;
use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::{
    default_types::{FrElement, FrField},
    pairing::BLS12381AtePairing,
};
use lambdaworks_math::polynomial::Polynomial;

type KZG = KateZaveruchaGoldberg<FrField, BLS12381AtePairing>;

// Load or generate SRS
let srs = StructuredReferenceString::from_file("srs.bin").unwrap();
let kzg = KZG::new(srs);

// Create a polynomial: p(x) = 1 + 2x + 3x^2
let p = Polynomial::new(&[
    FrElement::from(1),
    FrElement::from(2),
    FrElement::from(3),
]);

// Commit to the polynomial
let commitment = kzg.commit(&p);

// Evaluate at z = 5
let z = FrElement::from(5);
let y = p.evaluate(&z);  // p(5) = 1 + 10 + 75 = 86

// Generate proof
let proof = kzg.open(&z, &y, &p);

// Verify
let is_valid = kzg.verify(&z, &y, &commitment, &proof);
assert!(is_valid);
```

### Batch Operations

KZG supports efficient batch proofs for multiple polynomial evaluations:

```rust
// Multiple polynomials
let p0 = Polynomial::new(&[FrElement::from(42)]);
let p1 = Polynomial::new(&[FrElement::from(1), FrElement::from(2)]);

let c0 = kzg.commit(&p0);
let c1 = kzg.commit(&p1);

// Evaluate both at the same point
let z = FrElement::from(7);
let y0 = p0.evaluate(&z);
let y1 = p1.evaluate(&z);

// Random challenge for batching
let upsilon = FrElement::from(123);

// Generate batch proof
let batch_proof = kzg.open_batch(&z, &[y0.clone(), y1.clone()], &[p0, p1], &upsilon);

// Verify batch
let valid = kzg.verify_batch(&z, &[y0, y1], &[c0, c1], &batch_proof, &upsilon);
assert!(valid);
```

### SRS Management

The SRS can be serialized and loaded:

```rust
// Serialize to bytes
let bytes = srs.as_bytes();

// Load from bytes
let loaded_srs = StructuredReferenceString::deserialize(&bytes).unwrap();

// Load from file
let srs = StructuredReferenceString::from_file("path/to/srs.bin").unwrap();
```

## FRI Commitment Scheme

Fast Reed-Solomon Interactive Oracle Proof (FRI) is a polynomial commitment scheme that uses hashing instead of pairings. It is the foundation of STARK proofs.

### How FRI Works

FRI commits to a polynomial through its evaluations, building a Merkle tree of the values:

1. **Commit**: Evaluate the polynomial over a large domain (the "LDE domain") and commit to the evaluations via a Merkle tree.

2. **Fold**: Receive a random challenge $\alpha$ and compute a new polynomial of half the degree:
   $$p'(x) = \frac{p(x) + p(-x)}{2} + \alpha \cdot \frac{p(x) - p(-x)}{2x}$$

3. **Repeat**: Continue folding until the polynomial becomes a constant.

4. **Query**: The verifier requests openings at random positions, and the prover provides Merkle proofs.

### FRI in lambdaworks

FRI is integrated into the STARK prover:

```rust
use stark_platinum_prover::fri::fri_commitment::FriLayer;

// FRI is used internally by the STARK prover
// Users typically interact with the higher-level prove/verify API
```

The FRI parameters are configured through proof options:

```rust
use stark_platinum_prover::proof::options::ProofOptions;

let options = ProofOptions::new(
    32,    // Security bits
    4,     // Blowup factor (domain extension)
    8,     // FRI number of queries
    32,    // Coset offset
);
```

### FRI vs KZG Comparison

| Property | KZG | FRI |
|----------|-----|-----|
| **Assumption** | Discrete log + pairing | Collision-resistant hash |
| **Proof size** | O(1) | O(log^2 n) |
| **Prover time** | O(n log n) | O(n log n) |
| **Verifier time** | O(1) pairings | O(log^2 n) hashes |
| **Setup** | Trusted (or universal) | None |
| **Quantum resistance** | No | Yes (with proper hash) |

## Merkle Trees

Merkle trees are used by FRI and other commitment schemes to commit to vectors of values:

```rust
use lambdaworks_crypto::merkle_tree::merkle::MerkleTree;
use lambdaworks_crypto::merkle_tree::backends::field_element::FieldElementBackend;
use sha3::Keccak256;

type Backend = FieldElementBackend<Stark252PrimeField, Keccak256, 32>;

// Build tree from field elements
let values: Vec<FE> = (1..17).map(FE::from).collect();
let tree = MerkleTree::<Backend>::build(&values).unwrap();

// Get root commitment
let root = tree.root.clone();

// Generate inclusion proof
let proof = tree.get_proof_by_pos(5).unwrap();

// Verify proof
let is_valid = proof.verify::<Backend>(&root, 5, &values[5]);
assert!(is_valid);
```

### Merkle Proof Structure

A Merkle proof consists of the sibling hashes along the path from a leaf to the root:

```rust
// Proof contains: leaf value, authentication path, leaf index
pub struct MerkleProof<T> {
    pub merkle_path: Vec<T>,
}
```

## Fiat-Shamir Transform

Interactive protocols become non-interactive using the Fiat-Shamir heuristic. Random challenges are derived from a hash of the transcript:

```rust
use stark_platinum_prover::transcript::StoneProverTranscript;

let mut transcript = StoneProverTranscript::new(&[]);

// Append commitment to transcript
transcript.append_bytes(&commitment.to_bytes_be());

// Get challenge derived from transcript state
let challenge = transcript.sample_field_element();
```

The transcript accumulates all prover messages, ensuring the verifier can reproduce the same challenges.

## Pedersen Commitments

Pedersen commitments are used for committing to individual field elements (not polynomials):

$$C = m \cdot G + r \cdot H$$

where $m$ is the message, $r$ is randomness, and $G$, $H$ are elliptic curve points.

Properties:
1. **Perfectly hiding**: $C$ reveals nothing about $m$ (assuming $r$ is random).
2. **Computationally binding**: Cannot find $m' \neq m$ with the same commitment.
3. **Additively homomorphic**: $C_1 + C_2 = \text{Commit}(m_1 + m_2, r_1 + r_2)$.

## Commitment Scheme Trait

lambdaworks defines a common interface for commitment schemes:

```rust
pub trait IsCommitmentScheme<F: IsField> {
    type Commitment;

    fn commit(&self, p: &Polynomial<FieldElement<F>>) -> Self::Commitment;

    fn open(
        &self,
        x: &FieldElement<F>,
        y: &FieldElement<F>,
        p: &Polynomial<FieldElement<F>>,
    ) -> Self::Commitment;

    fn verify(
        &self,
        x: &FieldElement<F>,
        y: &FieldElement<F>,
        commitment: &Self::Commitment,
        proof: &Self::Commitment,
    ) -> bool;
}
```

This allows different commitment schemes to be used interchangeably.

## Security Considerations

1. **SRS Security (KZG)**: The SRS must be generated securely. If the secret $\tau$ is known, fake proofs can be created. Use a trusted setup ceremony or a universal/updatable SRS.

2. **Domain Size**: The polynomial degree must be less than the SRS size for KZG, or the domain size for FRI.

3. **Hash Function (FRI)**: Use a cryptographically secure hash function. FRI security relies on collision resistance.

4. **Soundness Error**: FRI has a small but non-zero probability of accepting invalid proofs. Increase the number of queries to reduce this.

## Further Reading

1. [KZG10 Paper](https://www.iacr.org/archive/asiacrypt2010/6477178/6477178.pdf) - Original KZG paper
2. [FRI Paper](https://eccc.weizmann.ac.il/report/2017/134/) - Fast Reed-Solomon IOPP
3. [EIP-4844](https://eips.ethereum.org/EIPS/eip-4844) - KZG for Ethereum blobs
4. [Multiproofs for KZG](https://dankradfeist.de/ethereum/2021/06/18/pcs-multiproofs.html) - Efficient batch verification
