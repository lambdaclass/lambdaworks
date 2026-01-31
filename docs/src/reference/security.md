# Security Considerations

This document outlines security considerations when using lambdaworks for cryptographic applications.

## General Principles

### 1. Use Established Primitives

lambdaworks provides implementations of well-studied cryptographic primitives. When building applications:

1. **Prefer standard constructions** over custom designs.
2. **Follow established protocols** rather than inventing new ones.
3. **Review security proofs** for the primitives you use.

### 2. Understand Your Threat Model

Before selecting primitives, understand:

1. What attacks are you defending against?
2. What is the expected attacker capability?
3. What are the consequences of a security breach?

## Field and Curve Security

### Field Size

The security level depends on field size:

| Field | Bits | Security Level |
|-------|------|----------------|
| Mersenne31 | 31 | ~15 bits (toy) |
| BabyBear | 31 | ~15 bits (toy) |
| Goldilocks | 64 | ~32 bits (low) |
| Stark252 | 252 | ~126 bits |
| BLS12-381 scalar | 255 | ~128 bits |
| BN254 scalar | 254 | ~128 bits |

**Recommendation**: For production systems, use fields with at least 128 bits of security.

### Small Field Extensions

When using small fields (BabyBear, Mersenne31), security comes from:

1. **Field extensions**: Using $\mathbb{F}_{p^k}$ instead of $\mathbb{F}_p$
2. **Multiple queries**: In FRI and other protocols

Ensure your configuration achieves the target security level.

### Elliptic Curve Security

For elliptic curve operations:

| Curve | Security Level | Notes |
|-------|----------------|-------|
| secp256k1 | ~128 bits | Not constant-time in lambdaworks |
| BLS12-381 | ~128 bits | Pairing-friendly |
| BN254 | ~128 bits | Ethereum compatible |
| Pallas/Vesta | ~128 bits | Cycle curves |

**Warning**: The secp256k1 implementation in lambdaworks is NOT constant-time. Do not use it for signing operations in production. Use a dedicated signing library.

## Proof System Security

### Trusted Setup (KZG, Groth16)

Systems using trusted setup require special care:

1. **Multi-party computation (MPC)**: Generate parameters using an MPC ceremony with multiple independent parties.

2. **Toxic waste destruction**: Ensure all parties securely delete their secret contributions.

3. **Verification**: Verify the output of setup ceremonies before use.

4. **SRS provenance**: Only use SRS from trusted sources (e.g., Powers of Tau ceremonies).

**If the trusted setup is compromised**, an attacker can forge proofs.

### STARK Soundness

STARK security depends on:

1. **Hash function security**: Use collision-resistant hash functions.
2. **Number of queries**: More FRI queries = lower soundness error.
3. **Blowup factor**: Higher blowup = better security margin.

Calculate the soundness error for your parameters:

$$\text{soundness error} \approx \left(\frac{1}{|F|}\right)^{\text{queries}} \cdot 2^{-\text{blowup}}$$

### Circuit Soundness

A bug in your circuit can make the entire proof system insecure:

1. **Under-constrained circuits**: Missing constraints allow invalid proofs.
2. **Over-constrained circuits**: May prevent valid witnesses from existing.
3. **Incorrect public inputs**: May leak private information.

**Best practices**:
1. Write comprehensive tests for your circuits.
2. Verify witnesses satisfy all constraints before proving.
3. Consider formal verification for critical circuits.
4. Get independent security audits.

## Random Number Generation

### For Cryptographic Operations

Use cryptographically secure random number generators:

```rust
use rand::rngs::OsRng;
use rand::RngCore;

let mut rng = OsRng;
let mut random_bytes = [0u8; 32];
rng.fill_bytes(&mut random_bytes);
```

**Never use**:
1. `rand::thread_rng()` for cryptographic purposes
2. Predictable seeds
3. Weak PRNGs

### Fiat-Shamir Security

The Fiat-Shamir transcript must include:

1. All public parameters
2. All prover messages
3. All commitments

Missing any component can enable attacks. The transcript hash function must be:

1. Collision-resistant
2. Preimage-resistant
3. Properly domain-separated

## Side-Channel Attacks

### Timing Attacks

Most lambdaworks operations are NOT constant-time:

1. **Field operations**: Variable-time multiplication and inversion
2. **Curve operations**: Non-constant-time scalar multiplication
3. **Conditional branches**: Data-dependent control flow

**For signing operations**, use dedicated constant-time libraries.

### Memory Access Patterns

Table lookups and memory access patterns may leak information through:

1. Cache timing attacks
2. Memory access side channels

## Implementation Bugs

### Known Limitations

1. **Error handling**: Some functions may panic on invalid input.
2. **Overflow checks**: Big integer operations may overflow if inputs exceed expected sizes.
3. **Serialization**: Malformed input may cause parsing errors or unexpected behavior.

### Input Validation

Always validate inputs:

```rust
// Validate field elements are in range
let element = FieldElement::from_bytes_be(&bytes)
    .map_err(|_| "Invalid field element")?;

// Validate curve points are on the curve and in subgroup
let point = Curve::create_point_from_affine(x, y)
    .map_err(|_| "Point not on curve")?;

// Validate proof components before verification
if proof.a.is_identity() || proof.b.is_identity() {
    return Err("Invalid proof component");
}
```

## Deployment Considerations

### WebAssembly

When deploying to WebAssembly:

1. **JavaScript interaction**: Ensure proper memory handling at the JS boundary.
2. **Random numbers**: Use `getrandom` with appropriate features for WASM.
3. **Performance**: WASM may be slower than native; adjust security parameters accordingly.

### no_std Environments

For embedded systems:

1. **Memory constraints**: Proof generation requires significant memory.
2. **Stack usage**: Deep recursion may cause stack overflow.
3. **Entropy**: Ensure access to a secure random source.

## Auditing and Testing

### Recommended Practices

1. **Fuzz testing**: Use fuzzing to find edge cases.

   ```bash
   cargo fuzz run <fuzzer_name>
   ```

2. **Property testing**: Use `proptest` for algebraic properties.

3. **Differential testing**: Compare against reference implementations.

4. **Security audits**: Engage professional auditors for production systems.

### Test Vectors

Use test vectors from:

1. Official specifications (e.g., EIP-4844 for KZG)
2. Reference implementations (e.g., blst for BLS12-381)
3. Cross-library verification

## Reporting Vulnerabilities

If you discover a security vulnerability in lambdaworks:

1. **Do not** disclose publicly before a fix is available.
2. **Report** via GitHub's security advisory feature or email security@lambdaclass.com.
3. **Provide** detailed reproduction steps.

See the [Security Policy](https://github.com/lambdaclass/lambdaworks/blob/main/.github/SECURITY.md) for more details.

## Summary Checklist

Before deploying lambdaworks in production:

- [ ] Field/curve provides adequate security level
- [ ] Trusted setup (if applicable) from reputable ceremony
- [ ] Circuit thoroughly tested and audited
- [ ] Proof parameters achieve target soundness
- [ ] Random number generation uses secure source
- [ ] Input validation on all external data
- [ ] No constant-time requirements violated
- [ ] WASM/no_std considerations addressed
- [ ] Security audit completed
