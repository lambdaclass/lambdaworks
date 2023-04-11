# Usage
Let's start with some examples.

At the moment our API supports the backend of PLONK. That is the all the setup, prove and verify algorithms. For the definition of a circuit and the creation of the $Q$ and $V$ matrices, as well as the execution of it to obtain the trace matrix $T$, we rely on external sources. We mainly use gnark temporarily for that purpose.

So to generate proofs and validate them, we need to feed the algorithms with precomputed values of the $Q$, $V$ and $T$ matrices, and the primitive root of unity $\omega$.

Let us see our API on a test circuit that provides all these values. The program in this case is the one that takes an input $x$, a private input $e$ and computes $y = xe +5$. As in the toy example of the recap, the output of the program is added to the public inputs and the circuit actually asserts that the output is the claimed value. So more precisely, the prover will generate a proof for the statement `ASSERT(x*e+5==y)`, where both $x,y$ are public inputs.

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

## Using gnark's frontend

### Exporting precomputed values from gnark's frontend
Here is a function written in `go` to use gnark's frontend and export a JSON file with all the precomputed values needed by our backend.

```go
type SerializedCircuit struct {
	N           int
	N_Padded    uint64
	Omega       string
	Input       []string
	Ql          []string
	Qr          []string
	Qm          []string
	Qo          []string
	Qc          []string
	A           []string
	B           []string
	C           []string
	Permutation []int64
}

func ToJSON(_r1cs *cs.SparseR1CS, pk *plonk_bls12381.ProvingKey, fullWitness fr_bls12381.Vector, witnessPublic fr_bls12381.Vector) {
	// n
	nbConstraints := len(_r1cs.Constraints)
	nbPublic := len(_r1cs.Public)
	n := nbConstraints + nbPublic
	omega := pk.Domain[0].Generator.Text(16)

	// Ql, Qm, Qr, Qo, Qk, S1, S2, S3
	var Ql, Qr, Qm, Qo, Qc []string

	for i := 0; i < nbPublic; i++ { 
		var minus_one fr_bls12381.Element
		minus_one = fr_bls12381.NewElement(1)
		minus_one.Neg(&minus_one)
		zero := fr_bls12381.NewElement(0)
		Ql = append(Ql, minus_one.Text(16))
		Qr = append(Qr, zero.Text(16))
		Qm = append(Qm, zero.Text(16))
		Qo = append(Qo, zero.Text(16))
		Qc = append(Qc, zero.Text(16))
	}

	for i := 0; i < nbConstraints; i++ { // constraints
		Ql = append(Ql, _r1cs.Coefficients[_r1cs.Constraints[i].L.CoeffID()].Text(16))
		Qr = append(Qr, _r1cs.Coefficients[_r1cs.Constraints[i].R.CoeffID()].Text(16))

		var new_Qm fr_bls12381.Element
		new_Qm.Set(&_r1cs.Coefficients[_r1cs.Constraints[i].M[0].CoeffID()]).Mul(&new_Qm, &_r1cs.Coefficients[_r1cs.Constraints[i].M[1].CoeffID()])

		Qm = append(Qm, new_Qm.Text(16))
		Qo = append(Qo, _r1cs.Coefficients[_r1cs.Constraints[i].O.CoeffID()].Text(16))
		Qc = append(Qc, _r1cs.Coefficients[_r1cs.Constraints[i].K].Text(16))
	}

	// Witness
	opt, _ := backend.NewProverConfig()
	var abc, _ = _r1cs.Solve(fullWitness, opt)
	var a, b, c []string
	for i := 0; i < len(_r1cs.Public); i++ {
		a = append(a, witnessPublic[i].Text(16))
		b = append(b, witnessPublic[0].Text(16))
		c = append(c, witnessPublic[0].Text(16))
	}
	for i := 0; i < nbConstraints; i++ { // constraints
		a = append(a, abc[_r1cs.Constraints[i].L.WireID()].Text(16))
		b = append(b, abc[_r1cs.Constraints[i].R.WireID()].Text(16))
		c = append(c, abc[_r1cs.Constraints[i].O.WireID()].Text(16))
	}

	var input []string
	for i := 0; i < len(_r1cs.Public); i++ {
		input = append(input, witnessPublic[i].Text(16))
	}

	data := SerializedCircuit{
		N:           n,
		Omega:       omega,
		N_Padded:    pk.Domain[0].Cardinality,
		Input:       input,
		Ql:          Ql,
		Qr:          Qr,
		Qm:          Qm,
		Qo:          Qo,
		Qc:          Qc,
		A:           a,
		B:           b,
		C:           c,
		Permutation: pk.Permutation,
	}
	file, _ := json.MarshalIndent(data, "", " ")
	_ = ioutil.WriteFile("frontend_precomputed_values.json", file, 0644)
}
```
#### Example
This is a simple example of how to use it using gnark's backend
```go
package main

import (
	"fmt"
	"log"

	"encoding/json"
	"io/ioutil"

	fr_bls12381 "github.com/consensys/gnark-crypto/ecc/bls12-381/fr"
	"github.com/consensys/gnark/backend"
	plonk_bls12381 "github.com/consensys/gnark/internal/backend/bls12-381/plonk"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/plonk"
	cs "github.com/consensys/gnark/constraint/bls12-381"
	"github.com/consensys/gnark/frontend/cs/scs"
	"github.com/consensys/gnark/test"

	"github.com/consensys/gnark/frontend"
)

// Circuit y == x**e
// only the bitSize least significant bits of e are used
type Circuit struct {
	// tagging a variable is optional
	// default uses variable name and secret visibility.
	X frontend.Variable `gnark:",public"`
	Y frontend.Variable `gnark:",public"`

	E frontend.Variable
}

// Define the circuit's constraints
func (circuit *Circuit) Define(api frontend.API) error {
	output := api.Mul(circuit.E, circuit.X)
	five := frontend.Variable(5)
	output2 := api.Add(output, five)
	api.AssertIsEqual(circuit.Y, output2)
	return nil
}

func main() {

	var circuit Circuit

	ccs, err := frontend.Compile(ecc.BLS12_381.ScalarField(), scs.NewBuilder, &circuit)
	if err != nil {
		fmt.Println("circuit compilation error")
	}

	_r1cs := ccs.(*cs.SparseR1CS)
	srs, err := test.NewKZGSRS(_r1cs)
	if err != nil {
		panic(err)
	}

	{
		var w Circuit
		w.X = 2
		w.E = 2
		w.Y = 9

		witnessFull, err := frontend.NewWitness(&w, ecc.BLS12_381.ScalarField())
		if err != nil {
			log.Fatal(err)
		}

		witnessPublic, err := frontend.NewWitness(&w, ecc.BLS12_381.ScalarField(), frontend.PublicOnly())
		if err != nil {
			log.Fatal(err)
		}

		pk, vk, err := plonk.Setup(ccs, srs)
		if err != nil {
			log.Fatal(err)
		}

		proof, err := plonk.Prove(ccs, pk, witnessFull)
		if err != nil {
			log.Fatal(err)
		}

		err = plonk.Verify(proof, vk, witnessPublic)
		if err != nil {
			log.Fatal(err)
		}

		fullWitness, _ := witnessFull.Vector().(fr_bls12381.Vector)
		publicWitness, _ := witnessPublic.Vector().(fr_bls12381.Vector)
		ToJSON(_r1cs, pk.(*plonk_bls12381.ProvingKey), fullWitness, publicWitness)
	}
}
```

## Importing precomputed values from JSON
To use the precomputed values exported from gnark's frontend there is a function `common_preprocessed_input_from_json` in the `test_utils` module that parses it and returns an instance of `Witness`, an instance of `CommonPreprocessedInput` and the public input array.

```rust
let json_string = fs::read_to_string(frontend_precomputed_values_json_filepath).unwrap();
let (witness, common_preprocessed_input, public_input) = common_preprocessed_input_from_json(&json_string);
let srs = test_srs(common_preprocessed_input.n);
let kzg = KZG::new(srs);
let verifying_key = setup(&common_preprocessed_input, &kzg);
let random_generator = TestRandomFieldGenerator {};
let prover = Prover::new(kzg.clone(), random_generator);
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
# Implementation details
In this section we discuss the implementation details of the plonk algorithm. We use the notation and terminology of the [recap](./recap.md) section. 

The implementation pretty much follows the rounds as are described in the recap. There are a few details that are worth mentioning.

## Commitment Scheme
The commitment scheme we use is the Kate-Zaverucha-Goldberg scheme with the `BLS 12 381` curve and the ate pairing.

The order $r$ of the cyclic subgroup is

```
0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001
```

The maximum power of two that divides $r-1$ is $2^{32}$. Therefore, that is the maximum possible order for a primitive root of unity in $\mathbb{F}_r$ with order a power of two. 

## Padding
All the matrices $Q, V, T, PI$ are padded with dummy rows so that their length is a power of two. To be able to interpolate their columns, we need a primitive root of unity $\omega$ of that order. That means that the maximum possible size for a circuit is $2^{32}$.

The entries of the dummy rows are filled in with zeroes in the $Q$, $V$ and $PI$ matrices. The $T$ matrix needs to be consistent with the $V$ matrix. Therefore it is filled with the value of the variable with index $0$. That is the first public input.

Some other rows in the $V$ matrix have also dummy values. These are the rows corresponding to the $B$ and $C$ columns of the public input rows. In the recap we denoted them with the empty `-` symbol. They are filled in with the same logic as the padding rows, as well as the corresponding values in the $T$ matrix.

