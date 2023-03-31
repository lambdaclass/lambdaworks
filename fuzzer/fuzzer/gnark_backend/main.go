package main

import "C"
import (
	
	"math/big"
	"reflect"

	"github.com/consensys/gnark/backend/plonk"
	"github.com/consensys/gnark/backend/witness"
	"github.com/consensys/gnark/frontend/schema"
)

// export Circuit
type Circuit struct {
	// tagging a variable is optional
	// default uses variable name and secret visibility.
	X frontend.Variable `gnark:",public"`
	E frontend.Variable `gnark:",public"`

	Y frontend.Variable
}

// export Define
func (circuit *Circuit) Define(api frontend.API) error {
	// specify constraints
    output := api.Mul(circuit.X, circuit.Y)

	api.AssertIsEqual(circuit.E, output)

	return nil
}

// export CreateCircuit
func CreateCircuit(privateInputX big.Int, privateInputY big.Int, publiInput big.Int)(circuit Circuit) {
	var w Circuit
		w.X = 2
		w.E = 2
		w.Y = 4

}

// export VerityWithCircuitWitness
func VerityWithCircuitWitness(circuit *Circuit, witness.Witness)() {

}

// export NewWitness
func NewWitness(assignment Circuit, field *big.Int, opts ...WitnessOption) (witness.Witness, error) {
	opt, err := options(opts...)
	if err != nil {
		return nil, err
	}

	// count the leaves
	s, err := schema.Walk(assignment, tVariable, nil)
	if err != nil {
		return nil, err
	}
	if opt.publicOnly {
		s.Secret = 0
	}

	// allocate the witness
	w, err := witness.New(field)
	if err != nil {
		return nil, err
	}

	// write the public | secret values in a chan
	chValues := make(chan any)
	go func() {
		defer close(chValues)
		schema.Walk(assignment, tVariable, func(leaf schema.LeafInfo, tValue reflect.Value) error {
			if leaf.Visibility == schema.Public {
				chValues <- tValue.Interface()
			}
			return nil
		})
		if !opt.publicOnly {
			schema.Walk(assignment, tVariable, func(leaf schema.LeafInfo, tValue reflect.Value) error {
				if leaf.Visibility == schema.Secret {
					chValues <- tValue.Interface()
				}
				return nil
			})
		}
	}()
	if err := w.Fill(s.Public, s.Secret, chValues); err != nil {
		return nil, err
	}

	return w, nil
}


