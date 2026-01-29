# ECDSA Signature Scheme

The Elliptic Curve Digital Signature Algorithm (ECDSA) is a widely-used digital signature scheme based on elliptic curve cryptography. It is used in many protocols including Bitcoin, Ethereum, and TLS.

## Disclaimer

This implementation is **NOT cryptographically secure** due to non-constant time operations. It should only be used for:
- Educational purposes
- Signature verification
- Testing and development

For production signing, use a constant-time implementation with proper side-channel protections.

## What is ECDSA?

ECDSA provides three fundamental properties:

- **Authentication**: Proves the identity of the signer
- **Integrity**: Ensures the message has not been altered
- **Non-repudiation**: The signer cannot deny having signed the message

## Protocol Overview

### Parameters

- An elliptic curve $E$ over a finite field (e.g., secp256k1)
- A base point $G$ on the curve with prime order $n$
- A cryptographic hash function $H$ (e.g., SHA-256)

### Key Generation

1. Choose a random private key $d \in [1, n-1]$
2. Compute the public key $Q = d \cdot G$

### Signing

To sign a message $m$:

1. Compute the message hash $z = H(m)$
2. Select a cryptographically random nonce $k \in [1, n-1]$
3. Compute the curve point $R = k \cdot G$
4. Compute $r = R_x \mod n$ (the x-coordinate of $R$)
5. Compute $s = k^{-1}(z + r \cdot d) \mod n$
6. The signature is $(r, s)$

**Critical**: The nonce $k$ must be:
- Truly random (use a CSPRNG)
- Unique for every signature
- Never reused or leaked

Reusing a nonce allows an attacker to recover the private key!

### Verification

To verify a signature $(r, s)$ on message $m$ with public key $Q$:

1. Verify $r, s \in [1, n-1]$
2. Compute $z = H(m)$
3. Compute $u_1 = z \cdot s^{-1} \mod n$
4. Compute $u_2 = r \cdot s^{-1} \mod n$
5. Compute $R' = u_1 \cdot G + u_2 \cdot Q$
6. The signature is valid if $R'_x \equiv r \pmod{n}$

## Implementation

This example uses the secp256k1 curve, the same curve used by Bitcoin and Ethereum.

### Usage

```rust
use ecdsa_signature::{sign, verify, derive_public_key, ScalarFE};

// Generate a private key (in practice, use a secure random generator)
let private_key = ScalarFE::from_hex_unchecked(
    "c9afa9d845ba75166b5c215767b1d6934e50c3db36e89b127b8a622b120f6721"
);

// Derive the public key
let public_key = derive_public_key(&private_key);

// Hash of the message to sign (use SHA-256 or similar)
let message_hash: [u8; 32] = /* SHA256(message) */;

// Generate a random nonce (CRITICAL: must be unique and random!)
let nonce = ScalarFE::from_hex_unchecked(
    "a6e3c57dd01abe90086538398355dd4c3b17aa873382b0f24d6129493d8aad60"
);

// Sign the message
let signature = sign(&message_hash, &private_key, &nonce)
    .expect("Signing failed");

// Verify the signature
assert!(verify(&message_hash, &signature, &public_key).is_ok());
```

## Security Considerations

### Nonce Generation

The security of ECDSA critically depends on proper nonce generation:

1. **RFC 6979**: For deterministic nonce generation, implement RFC 6979 which derives $k$ from the private key and message hash using HMAC-DRBG.

2. **Random Nonces**: If using random nonces, ensure they come from a cryptographically secure random number generator (CSPRNG).

3. **Never Reuse**: Each signature must use a unique nonce. Reusing a nonce with different messages allows private key recovery.

### Side-Channel Attacks

This implementation does not protect against:
- Timing attacks
- Power analysis
- Cache timing attacks

Production implementations must use constant-time operations for all secret-dependent computations.

## References

- [SEC 1: Elliptic Curve Cryptography](https://www.secg.org/sec1-v2.pdf)
- [RFC 6979: Deterministic ECDSA](https://tools.ietf.org/html/rfc6979)
- [FIPS 186-4: Digital Signature Standard](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.186-4.pdf)
