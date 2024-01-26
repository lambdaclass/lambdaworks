# lambdaworks Provers

Provers allow one party, the prover, to show to other parties, the verifiers, that a given computer program has been executed correctly by means of a cryptographic proof. This proof ideally satisfies the following two properties: it is fast to verify and its size is small (smaller than the size of the witness). All provers have a `prove` function, which takes some description of the program and other input and outputs a proof. There is also a `verify` function which takes the proof and other input and accepts or rejects the proof.

This folder contains the different provers currently supported by lambdaworks:
- [Groth 16](https://github.com/lambdaclass/lambdaworks/tree/main/provers/groth16)
- [Plonk](https://github.com/lambdaclass/lambdaworks/tree/main/provers/plonk)
- [STARKs](https://github.com/lambdaclass/lambdaworks/tree/main/provers/stark)
- [Cairo](https://github.com/lambdaclass/lambdaworks/tree/main/provers/cairo)

The reference papers for each of the provers is given below:
- [Groth 16](https://eprint.iacr.org/2016/260)
- [Plonk](https://eprint.iacr.org/2019/953)
- [STARKs](https://eprint.iacr.org/2018/046.pdf)

A brief description of the Plonk and STARKs provers can be found [here](https://github.com/lambdaclass/lambdaworks/tree/main/docs/src)

Using one prover or another depends on usecase and other desired properties. We recommend reading and understanding how each prover works, so as to choose the most adequate.
- Groth 16: Shortest proof length. Security depends on pairing-friendly elliptic curves. Needs a new trusted setup for every program you want to prove.
- Plonk (using KZG as commitment scheme): Short proof length. Security depends on pairing-friendly elliptic curves. Universal trusted setup.
- STARKs: longer proof length. Security depends on collision-resistant hash functions. Conjectured to be post-quantum secure. Transparent (no trusted setup).

## Using provers

- [Cairo prover](https://github.com/lambdaclass/lambdaworks/blob/main/provers/cairo/README.md)
- [Plonk prover](https://github.com/lambdaclass/lambdaworks/blob/main/provers/plonk/README.md)
