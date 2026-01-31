# Glossary

This glossary defines key terms used throughout lambdaworks and zero-knowledge cryptography.

## A

### AIR (Algebraic Intermediate Representation)
A way of expressing computations as polynomial constraints. In STARKs, the AIR defines transition constraints (relating consecutive states) and boundary constraints (fixing values at specific positions).

### Affine Coordinates
The standard representation of elliptic curve points as $(x, y)$ pairs. Compare with projective coordinates, which add a third coordinate to avoid divisions.

## B

### Blowup Factor
In STARKs, the ratio between the evaluation domain size and the trace length. Higher blowup factors increase security but also proof size. Typical values are 4, 8, or 16.

### Boundary Constraints
Constraints in an AIR that fix the values of certain cells in the execution trace. For example, specifying initial values or final outputs.

### BLS12-381
A pairing-friendly elliptic curve widely used in cryptography (Ethereum 2.0, Zcash). Provides approximately 128 bits of security.

### BN254
A pairing-friendly elliptic curve used in Ethereum for precompiled contracts. Also known as alt_bn128.

## C

### Commitment Scheme
A cryptographic primitive allowing one to commit to a value while keeping it hidden, then later reveal it. Properties: binding (cannot change the committed value) and hiding (commitment reveals nothing).

### Cofactor
For an elliptic curve group, the ratio between the full group order and the prime subgroup order. Security requires working in the prime-order subgroup.

### Coset
In FFT-based proofs, a shifted copy of a multiplicative subgroup. Used to evaluate polynomials at points distinct from the trace domain.

## D

### DEEP Method
Technique in STARK proofs for securely combining multiple polynomial commitments. Samples a random point outside the evaluation domain.

### Degree (Polynomial)
The highest power of the variable with a non-zero coefficient. A polynomial $3x^5 + 2x + 1$ has degree 5.

## E

### Execution Trace
In STARKs, a matrix representing the computation. Each column is a register, each row is a state, and transitions between rows follow the AIR constraints.

### Extension Field
A larger field containing a base field. For example, the complex numbers extend the reals. Used in STARKs for security amplification.

## F

### FFT (Fast Fourier Transform)
An algorithm to evaluate a polynomial at all roots of unity in $O(n \log n)$ time. Essential for efficient polynomial manipulation.

### FFT-Friendly Field
A prime field where the multiplicative group order is divisible by a large power of 2, enabling efficient FFT. Examples: Stark252, BabyBear, Goldilocks.

### Field Element
A member of a finite field. In lambdaworks, represented by `FieldElement<F>` where `F` is the field type.

### FRI (Fast Reed-Solomon IOP)
A protocol proving that a function is close to a low-degree polynomial. Used in STARKs for polynomial commitments without pairings.

### Fiat-Shamir Transform
A technique to convert interactive proofs to non-interactive ones by deriving challenges from a hash of the transcript.

## G

### Generator (Curve)
A point $G$ on an elliptic curve such that all points in the (sub)group can be expressed as $nG$ for some integer $n$.

### GKR Protocol
An efficient interactive proof protocol for layered arithmetic circuits. Combined with sumcheck, enables fast verification.

### Groth16
A zk-SNARK producing the smallest proofs (~200 bytes). Requires per-circuit trusted setup.

## I

### Interpolation
The process of finding a polynomial that passes through given points. For $n$ points, a unique polynomial of degree at most $n-1$ exists.

## K

### KZG Commitment
Kate-Zaverucha-Goldberg polynomial commitment scheme using elliptic curve pairings. Produces constant-size commitments and proofs.

## L

### Lagrange Basis
Polynomials $L_i(x)$ that equal 1 at $x_i$ and 0 at other interpolation points. Any polynomial can be written as a linear combination of Lagrange basis polynomials.

### LDE (Low Degree Extension)
Extending a polynomial's evaluations from a small domain to a larger one. Used in STARKs to create redundancy for error detection.

## M

### Merkle Tree
A tree of hash values where each non-leaf node is the hash of its children. Enables efficient proofs of inclusion.

### Montgomery Form
A representation of field elements as $aR \mod p$ instead of $a \mod p$, enabling faster modular multiplication by avoiding division.

### MSM (Multi-Scalar Multiplication)
Computing $\sum_i s_i P_i$ for scalars $s_i$ and elliptic curve points $P_i$. A performance-critical operation in proof systems.

## N

### NTT (Number Theoretic Transform)
The FFT applied to finite fields. Used interchangeably with FFT in the context of lambdaworks.

## P

### Pairing
A bilinear map $e: G_1 \times G_2 \rightarrow G_T$ on elliptic curves. Enables constructions like BLS signatures and KZG commitments.

### PLONK
A universal SNARK using a single trusted setup for all circuits up to a given size. Produces ~1KB proofs.

### Polynomial Commitment
A commitment to a polynomial enabling proofs about its evaluations without revealing the full polynomial.

### Prime Field
A finite field with a prime number of elements. All non-zero elements have multiplicative inverses.

### Projective Coordinates
Representing elliptic curve points as $(X : Y : Z)$ where affine coordinates are $(X/Z, Y/Z)$. Avoids expensive field inversions during point operations.

### Prover
The party generating a proof. Must know the witness (private inputs) to create a valid proof.

## Q

### QAP (Quadratic Arithmetic Program)
A polynomial representation of R1CS constraints. Used in Groth16 and other SNARKs.

## R

### R1CS (Rank-1 Constraint System)
A constraint format where each constraint has the form $(a \cdot s)(b \cdot s) = c \cdot s$ for witness vector $s$ and sparse vectors $a, b, c$.

### Roots of Unity
Elements $\omega$ such that $\omega^n = 1$. The $n$-th roots of unity form a multiplicative group of order $n$.

## S

### Scalar Field
The field of scalars for an elliptic curve group, with order equal to the number of points in the (sub)group.

### SRS (Structured Reference String)
The public parameters for a proof system, generated during trusted setup. Contains commitments to powers of a secret.

### SNARK (Succinct Non-interactive ARgument of Knowledge)
A proof system producing short proofs (succinct) that can be verified without interaction (non-interactive) and prove knowledge of a witness.

### STARK (Scalable Transparent ARgument of Knowledge)
A proof system requiring no trusted setup (transparent) with proofs that scale polylogarithmically with computation size.

### Sumcheck Protocol
An interactive protocol proving the sum of a multivariate polynomial over the Boolean hypercube. A building block for GKR and other protocols.

## T

### Trace (Execution Trace)
See Execution Trace.

### Transcript
In Fiat-Shamir, the record of all messages exchanged. Used to derive random challenges.

### Transition Constraints
Constraints relating consecutive rows of an execution trace. Define how the computation evolves step by step.

### Trusted Setup
A ceremony generating public parameters containing hidden randomness ("toxic waste"). If the randomness is compromised, fake proofs can be created.

## V

### Vanishing Polynomial
A polynomial that is zero at all points in a specified set. For a set $H$, the vanishing polynomial is $Z_H(x) = \prod_{h \in H}(x - h)$.

### Verifier
The party checking a proof. Should be efficient (much faster than re-executing the computation).

## W

### Witness
The private inputs to a proof. The prover knows the witness; the verifier only sees public inputs.

## Z

### Zero-Knowledge
The property that a proof reveals nothing beyond the truth of the statement being proved.

### zk-SNARK
A SNARK with the additional property of zero-knowledge, hiding the witness from the verifier.

### zk-STARK
A STARK with zero-knowledge properties.
