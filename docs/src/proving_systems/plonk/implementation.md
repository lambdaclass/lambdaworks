# Implementation
In this section we discuss the implementation details of the plonk algorithm. Previous knowledge of the algorithm is assumed. To read more about it, you can see the section "Recap". We'll discuss in order the steps of the algorithm: setup, prover and verifier.

In the first place a Field, a Polynomial Commitment Scheme and a `RandomNumber` generator are required.

## Setup
We first need some structs to store the info needed for the algorithm:
- `CommonPreprocessedInput`: program structure (polynomials encoding the constraints and copy constraints) and domain of the interpolated polynomials.
- `VerificationKey`: stores the commitment for the polynomials, so that the verifier can assert that the prover used the correct program.

At this point the only things needed are a `Field` to know the coefficients of the polynomials and a polynomial commitment scheme to generate the `VerificationKey`.

The `k1` and `k2` parameters are chosen at these step. These parameters are chosen as `ROOT_R_MINUS_ONE_OF_UNITY` and `ROOT_P_OF_UNITY`.

Also, the first constraints in the polynomial refer to the public input. The `QL` polynomial has minus ones in the first K constraints (where K is the number of public inputs), and the prover then fills the public input with the `PI` polynomial.

## Prover
We have the following structs:

- `Witness`: stores the assignment to the variables taken from the program execution (trace).
- `Proof`:
- `Prover`:

The result of each round is stored in `RoundResult` structs. Lets go round by round and see what things are important to consider.

### Round 1
This round commits the witness polynomials. Remember that z_h, the polynomial that has roots over the domain, can also be written as $$. There's a specific function to blind the polynomials.

## Round 2
This is basically the equation found in the paper. To blind the polynomials

## Round 3
- Difference between degrees of t_lo, t_mid and t_hi. These polynomials are not blinded in Gnark.

## Round 4
Just the evaluations of the polynomials at the challenge zeta.

## Round 5
The `L1` polynomial is not computed exactly the same. A different polynomial with the same properties its used.

The batch commitment uses different polynomials (e.g.: linearized polynomial).

## Verifier
The goal is to reconstruct 

# Examples
## Creating a proof
```rust=
// This is the circuit for x * e + 5 == y
let common_preprocessed_input = test_common_preprocessed_input_2();
let srs = test_srs(common_preprocessed_input.n);

// Public input
let x = FieldElement::from(2_u64);
let y = FieldElement::from(11_u64);

// Private variable
let e = FieldElement::from(3_u64);

let public_input = vec![x.clone(), y];
let witness = test_witness_2(x, e);

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

## Exporting a circuit from GNark as JSON

```go
// Copyright 2020 ConsenSys AG
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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

// In this example we show how to use PLONK with KZG commitments. The circuit that is
// showed here is the same as in ../exponentiate.

// Circuit y == x**e
// only the bitSize least significant bits of e are used
type Circuit struct {
	// tagging a variable is optional
	// default uses variable name and secret visibility.
	X frontend.Variable `gnark:",public"`
	Y frontend.Variable `gnark:",public"`

	E frontend.Variable
}

// Define declares the circuit's constraints
func (circuit *Circuit) Define(api frontend.API) error {
	output := api.Mul(circuit.E, circuit.X)
	five := frontend.Variable(5)
	output2 := api.Add(output, five)
	api.AssertIsEqual(circuit.Y, output2)
	return nil
}

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

	// Correct data: the proof passes
	{
		// Witnesses instantiation. Witness is known only by the prover,
		// while public w is a public data known by the verifier.
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
		//_, err := plonk.Setup(r1cs, kate, &publicWitness)
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

func ToJSON(_r1cs *cs.SparseR1CS, pk *plonk_bls12381.ProvingKey, fullWitness fr_bls12381.Vector, witnessPublic fr_bls12381.Vector) {
	// n
	nbConstraints := len(_r1cs.Constraints)
	nbPublic := len(_r1cs.Public)
	n := nbConstraints + nbPublic
	omega := pk.Domain[0].Generator.Text(16)

	// Ql, Qm, Qr, Qo, Qk, S1, S2, S3
	var Ql, Qr, Qm, Qo, Qc []string

	for i := 0; i < nbPublic; i++ { // placeholders (-PUB_INPUT_i + qk_i = 0) TODO should return error is size is inconsistant
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
	//fullWitness, _ := witnessFull.Vector().(fr_bls12381.Vector)
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
	_ = ioutil.WriteFile("test.json", file, 0644)
}
```

## Importing circuit from JSON

