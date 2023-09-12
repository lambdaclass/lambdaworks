# Implementation
In this section we discuss the implementation details of the PLONK algorithm. We use the notation and terminology of the [protocol](./protocol.md) and [recap](./recap.md) sections. 

At the moment our API supports the backend of PLONK. That is, all the setup, prove and verify algorithms. We temporarily rely on external sources for the definition of a circuit and the creation of the $Q$ and $V$ matrices, as well as the execution of it to obtain the trace matrix $T$. We mainly use gnark temporarily for that purpose.

So to generate proofs and validate them, we need to feed the algorithms with precomputed values of the $Q$, $V$ and $T$ matrices, and the primitive root of unity $\omega$.

Let us see our API on a test circuit that provides all these values. The program in this case is the one that takes an input $x$, a private input $e$ and computes $y = xe +5$. As in the toy example of the recap, the output of the program is added to the public inputs and the circuit actually asserts that the output is the claimed value. So more precisely, the prover will generate a proof for the statement `ASSERT(x*e+5==y)`, where both $x,y$ are public inputs.
# Usage
Here is the happy path.

```rust
// This is the common preprocessed input for
// the test circuit ( ASSERT(x * e + 5 == y) )
let common_preprocessed_input = test_common_preprocessed_input_2();

// Input
let x = FieldElement::from(2_u64);

// Private input
let e = FieldElement::from(3_u64);

let y, witness = test_witness_2(x, e);

let srs = test_srs(common_preprocessed_input.n);
let kzg = KZG::new(srs);

let verifying_key = setup(&common_preprocessed_input, &kzg);

let random_generator = TestRandomFieldGenerator {};
let prover = Prover::new(kzg.clone(), random_generator);

let public_input = vec![x.clone(), y];

let proof = prover.prove(
    &witness,
    &public_input,
    &common_preprocessed_input,
    &verifying_key,
);

let verifier = Verifier::new(kzg);
assert!(verifier.verify(
    &proof,
    &public_input,
    &common_preprocessed_input,
    &verifying_key
));
```

Let's brake it down. The helper function `test_common_preprocessed_input_2()` returns an instance of the following struct for the particular test circuit:
```rust
pub struct CommonPreprocessedInput<F: IsField> {
    pub n: usize,
    pub domain: Vec<FieldElement<F>>,
    pub omega: FieldElement<F>,
    pub k1: FieldElement<F>,

    pub ql: Polynomial<FieldElement<F>>,
    pub qr: Polynomial<FieldElement<F>>,
    pub qo: Polynomial<FieldElement<F>>,
    pub qm: Polynomial<FieldElement<F>>,
    pub qc: Polynomial<FieldElement<F>>,

    pub s1: Polynomial<FieldElement<F>>,
    pub s2: Polynomial<FieldElement<F>>,
    pub s3: Polynomial<FieldElement<F>>,

    pub s1_lagrange: Vec<FieldElement<F>>,
    pub s2_lagrange: Vec<FieldElement<F>>,
    pub s3_lagrange: Vec<FieldElement<F>>,
}
```
Apart from the eight polynomials in the canonical basis, we store also here the number of constraints $n$, the domain $H$, the primitive $n$-th of unity $\omega$ and the element $k_1$. The element $k_2$ will be $k_1^2$. For convenience, we also store the polynomials $S_{\sigma i}$ in Lagrange form.

The following lines define the particular values of the program input $x$ and the private input $e$.
```rust
// Input
let x = FieldElement::from(2_u64);

// Private input
let e = FieldElement::from(3_u64);
let y, witness = test_witness_2(x, e);
```
 The function `test_witness_2(x, e)` returns an instance of the following struct, that holds the polynomials that interpolate the columns $A, B, C$ of the trace matrix $T$.
```rust
pub struct Witness<F: IsField> {
    pub a: Vec<FieldElement<F>>,
    pub b: Vec<FieldElement<F>>,
    pub c: Vec<FieldElement<F>>,
}
```
Next the commitment scheme KZG (Kate-Zaverucha-Goldberg) is instantiated.
```rust
let srs = test_srs(common_preprocessed_input.n);
let kzg = KZG::new(srs);
```
The `setup` function performs the setup phase. It only needs the common preprocessed input and the commitment scheme.
```rust
let verifying_key = setup(&common_preprocessed_input, &kzg);
```
It outputs an instance of the struct `VerificationKey`.
```rust
pub struct VerificationKey<G1Point> {
    pub qm_1: G1Point,
    pub ql_1: G1Point,
    pub qr_1: G1Point,
    pub qo_1: G1Point,
    pub qc_1: G1Point,

    pub s1_1: G1Point,
    pub s2_1: G1Point,
    pub s3_1: G1Point,
}
```
It stores the commitments of the eight polynomials of the common preprocessed input. The suffix `_1` means it is a commitment. It comes from the notation $[f]_1$, where $f$ is a polynomial.

Then a prover is instantiated
```rust
let random_generator = TestRandomFieldGenerator {};
let prover = Prover::new(kzg.clone(), random_generator);
```
The prover is an instance of the struct `Prover`:
```rust
pub struct Prover<F, CS, R>
where
  F:  IsField,
  CS: IsCommitmentScheme<F>,
  R:  IsRandomFieldElementGenerator<F>
  {
    commitment_scheme: CS,
    random_generator: R,
    phantom: PhantomData<F>,
}
```
It stores an instance of a commitment scheme and a random field element generator needed for blinding polynomials.

Then the public input is defined. As we mentioned in the recap, the public input contains the output of the program.
```rust
let public_input = vec![x.clone(), y];
```

We then generate a proof using the prover's method `prove`
```rust
let proof = prover.prove(
    &witness,
    &public_input,
    &common_preprocessed_input,
    &verifying_key,
);
```
The output is an instance of the struct `Proof`.
```rust
pub struct Proof<F: IsField, CS: IsCommitmentScheme<F>> {
    // Round 1.
    /// Commitment to the wire polynomial `a(x)`
    pub a_1: CS::Commitment,
    /// Commitment to the wire polynomial `b(x)`
    pub b_1: CS::Commitment,
    /// Commitment to the wire polynomial `c(x)`
    pub c_1: CS::Commitment,

    // Round 2.
    /// Commitment to the copy constraints polynomial `z(x)`
    pub z_1: CS::Commitment,

    // Round 3.
    /// Commitment to the low part of the quotient polynomial t(X)
    pub t_lo_1: CS::Commitment,
    /// Commitment to the middle part of the quotient polynomial t(X)
    pub t_mid_1: CS::Commitment,
    /// Commitment to the high part of the quotient polynomial t(X)
    pub t_hi_1: CS::Commitment,

    // Round 4.
    /// Value of `a(ζ)`.
    pub a_zeta: FieldElement<F>,
    /// Value of `b(ζ)`.
    pub b_zeta: FieldElement<F>,
    /// Value of `c(ζ)`.
    pub c_zeta: FieldElement<F>,
    /// Value of `S_σ1(ζ)`.
    pub s1_zeta: FieldElement<F>,
    /// Value of `S_σ2(ζ)`.
    pub s2_zeta: FieldElement<F>,
    /// Value of `z(ζω)`.
    pub z_zeta_omega: FieldElement<F>,

    // Round 5
    /// Value of `p_non_constant(ζ)`.
    pub p_non_constant_zeta: FieldElement<F>,
    ///  Value of `t(ζ)`.
    pub t_zeta: FieldElement<F>,
    /// Batch opening proof for all the evaluations at ζ
    pub w_zeta_1: CS::Commitment,
    /// Single opening proof for `z(ζω)`.
    pub w_zeta_omega_1: CS::Commitment,
}
```

Finally, we instantiate a verifier.
```rust
let verifier = Verifier::new(kzg);
```

It's an instance of `Verifier`:
```rust
struct Verifier<F: IsField, CS: IsCommitmentScheme<F>> {
    commitment_scheme: CS,
    phantom: PhantomData<F>,
}
```

Finally, we call the verifier's method `verify` that outputs a `bool`.
```rust
assert!(verifier.verify(
    &proof,
    &public_input,
    &common_preprocessed_input,
    &verifying_key
));
```

## Padding
All the matrices $Q, V, T, PI$ are padded with dummy rows so that their length is a power of two. To be able to interpolate their columns, we need a primitive root of unity $\omega$ of that order. Given the particular field used in our implementation, that means that the maximum possible size for a circuit is $2^{32}$.

The entries of the dummy rows are filled in with zeroes in the $Q$, $V$ and $PI$ matrices. The $T$ matrix needs to be consistent with the $V$ matrix. Therefore it is filled with the value of the variable with index $0$.

Some other rows in the $V$ matrix have also dummy values. These are the rows corresponding to the $B$ and $C$ columns of the public input rows. In the recap we denoted them with the empty `-` symbol. They are filled in with the same logic as the padding rows, as well as the corresponding values in the $T$ matrix.

# Implementation details

The implementation pretty much follows the rounds as are described in the [protocol](./protocol.md) section. There are a few details that are worth mentioning.

## Commitment Scheme
The commitment scheme we use is the [Kate-Zaverucha-Goldberg](https://www.iacr.org/archive/asiacrypt2010/6477178/6477178.pdf) scheme with the `BLS 12 381` curve and the ate pairing. It can be found in the `commitments` module of the `lambdaworks_crypto` package.

The order $r$ of the cyclic subgroup is

```
0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
```

The maximum power of two that divides $r-1$ is $2^{32}$. Therefore, that is the maximum possible order for a primitive root of unity in $\mathbb{F}_r$ with order a power of two. 

## Fiat-Shamir

### Transcript strategy

Here we describe our implementation of the transcript used for the Fiat-Shamir heuristic.

A `Transcript` exposes two methods: `append` and `challenge`.

The method `append` adds a message to the transcript by updating the internal state of the hasher with the raw bytes of the message. 

The  method `challenge` returns the result of the hasher using the current internal state of the hasher. It subsequently resets the hasher and updates the internal state with the last result.

Here is an example of this process:

1. Start a fresh transcript. 
2. Call `append` and pass `message_1`.
3. Call `append` and pass `message_2`.
4. The internal state of the hasher at this point is `message_2 || message_1`.
5. Call `challenge`. The output is `Hash(message_2 || message_1)`.
6. Call `append` and pass `message_3`.
7. Call `challenge`. The output is `Hash(message_3 || Hash(message_2 || message_1))`.
8. Call `append` and pass `message_4`.

The internal state of the hasher at the end of this exercise is `message_4 || Hash(message_3 || Hash(message_2 || message_1))`

The underlying hasher function we use is `h=sha3`.

### Field elements
The result of every challenge is a $256$-bit string, which is interpreted as an integer in big-endian order. A field element is constructed out of it by taking modulo the field order. The prime field used in this implementation has a $255$-bit order. Therefore some field elements are more probable to occur than others because they have more representatives as 256-bit integers.

### Strong Fiat-Shamir
The first messages added to the transcript are all commitments of the polynomials of the common preprocessed input and the values of the public inputs. This prevents a known vulnerability called "weak Fiat-Shamir".
Check out the following resources to learn more about it.

- [What can go wrong (zkdocs)](https://www.zkdocs.com/docs/zkdocs/protocol-primitives/fiat-shamir/#what-can-go-wrong)
- [How not to Prove Yourself: Pitfalls of the Fiat-Shamir Heuristic and Applications to Helios](https://eprint.iacr.org/2016/771.pdf)
- [Weak Fiat-Shamir Attacks on Modern Proof Systems](https://eprint.iacr.org/2023/691)
