# LambdaWorks

<div>

[![Telegram Chat][tg-badge]][tg-url]

[tg-badge]: https://img.shields.io/static/v1?color=green&logo=telegram&label=chat&style=flat&message=join
[tg-url]: https://t.me/+6jzrsK6qR7w3MTkx

</div>

The library for kids who wanna learn how to do SNARKs and learn other cryptographic stuff too.

References:
- [arkworks-rs](https://github.com/arkworks-rs/)
- [gnark](https://github.com/ConsenSys/gnark)

## Building blocks

- Finite Field Algebra
- Elliptic curve models
- Elliptic curve operations
- Arithmetization schemes
- Polynomial commitment schemes
- PIOP
- Cryptographic tools
- Advanced tools: agreggation, recursion, accumulation
- Protocols
- Gadgets

## Blocks
### Finite Field Algebra
- Big integer representation
- Basic algebra: addition, multiplication, subtraction, inversion, square root (Tonelli–Shanks)
- Field extensions
- Number theoretic transform
- Polynomial operations
- Fast Fourier Transform
- Montgomery and Barrett

### Elliptic curve models
- BLS12-381 (H)
- BLS12-377 (H)
- secp256k1 (H)
- Ed25519 (H)
- Jubjub (M)
- BN254 (M)
- Pasta: Pallas and Vesta (L)
- Forms:
1. Affine (H)
2. Projective (H)
3. Montgomery (M)
4. Twisted Edwards (H)
5. Jacobi (L)

### Elliptic curve operations
- Add, double, scalar multiplication.
- Multiscalar multiplication (Pippenger)
- Weyl, Tate and Ate pairings.

### Arithmetization
- R1CS - gadgets (H)
- AIR (M)
- Plonkish (H)
- ACIR (L)

### Polynomial commitment schemes
- KZG and variants
- Hashing
- Inner product arguments

### PIOP/PCS
- Groth16
- Plonk
- Marlin
- FRI

### [Crypto primitives](https://github.com/RustCrypto)
- Pseudorandom generator
- Hashes
- Blake2
- Keccak
- Poseidon
- Pedersen
- Encryption schemes
- AES
- ChaCha20
- Rescue
- ElGamal

### Protocol
- Fiat-Shamir

### Gadgets
