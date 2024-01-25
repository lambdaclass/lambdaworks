# lambdaworks Provers

Provers allow one party, the prover, to show to other parties, the verifiers, that a given computer program has been executed correctly by means of a cryptographic proof. This proof ideally satisfies the following two properties: it is fast to verify and its size is small (smaller than the size of the witness). 

This folder contains the different provers currently supported by lambdaworks:
- Groth 16
- Plonk
- STARKs
- Cairo

The reference papers for each of the provers is given below:
- [Groth 16](https://eprint.iacr.org/2016/260)
- [Plonk](https://eprint.iacr.org/2019/953)
- [STARKs](https://eprint.iacr.org/2018/046.pdf)

A brief description of the Plonk and STARKs provers can be found [here](https://github.com/lambdaclass/lambdaworks/tree/main/docs/src)
