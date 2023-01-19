# LambdaWorks

# Building blocks

- Finite Field Algebra
- Elliptic curve models
- Elliptic curve operations
- Arithmetization schemes
- Polynomial commitment schemes
- PIOP
- Cryptographic tools
- Advanced tools: agreggation, recursion, accumulation

## Submodules

### Finite Field Algebra
- Big integer representation
- Basic algebra: addition, multiplication, subtraction, inversion, square root.
- Field extensions.
- Number theoretic transform
- Polynomial operations

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

### Crypto primitives (https://github.com/RustCrypto)
- Pseudorandom generator
- Hashes
1. Blake2
2. Keccak
3. Poseidon
4. Pedersen
- Encryption schemes
5. AES
6. ChaCha20
7. Rescue
8. ElGamal
