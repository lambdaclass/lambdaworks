# RSA Algorithm Implementation

This is an implementation of the RSA (Rivest-Shamir-Adleman) cryptographic algorithm in Rust. RSA is one of the first public-key cryptosystems and is widely used for secure data transmission.

## Table of Contents
- [Overview](#overview)
- [Mathematical Background](#mathematical-background)
- [Implementation Details](#implementation-details)
- [Usage Examples](#usage-examples)
- [Security Considerations](#security-considerations)
- [Performance Considerations](#performance-considerations)

## Overview

RSA is an asymmetric cryptographic algorithm that uses a pair of keys:
- **Public key**: Used for encryption and is shared openly
- **Private key**: Used for decryption and must be kept secret

The security of RSA relies on the practical difficulty of factoring the product of two large prime numbers.

## Mathematical Background

### Key Generation

1. Choose two distinct prime numbers `p` and `q`
2. Compute `n = p * q`
3. Calculate Euler's totient function: `φ(n) = (p-1) * (q-1)`
4. Choose an integer `e` such that `1 < e < φ(n)` and `gcd(e, φ(n)) = 1`
5. Compute `d` such that `d * e ≡ 1 (mod φ(n))`

The public key is `(e, n)` and the private key is `d`.

### Euler's Totient Function

Euler's totient function `φ(n)` counts the positive integers up to `n` that are coprime to `n`. For a prime number `p`, `φ(p) = p-1` since all numbers less than `p` are coprime to `p`.

For a product of two primes `n = p * q`, `φ(n) = φ(p) * φ(q) = (p-1) * (q-1)`.

### Encryption and Decryption

- **Encryption**: `c = m^e mod n` where `m` is the message and `c` is the ciphertext
- **Decryption**: `m = c^d mod n`

## Implementation Details

This implementation includes:

1. **Basic RSA operations**:
   - Key generation from two prime numbers
   - Encryption and decryption of numeric values

2. **PKCS#1 v1.5 padding**:
   - For secure encryption of arbitrary byte data
   - Padding scheme: `0x00 || 0x02 || PS || 0x00 || M`
   - Where PS is a sequence of non-zero random bytes

3. **Modular inverse calculation**:
   - Using the Extended Euclidean Algorithm

## Usage Examples

### Basic Numeric Example

```rust
use rsa::RSA;
use num_bigint::BigUint;
use num_traits::FromPrimitive;

// Create an RSA instance with small primes (for demonstration only)
let p = BigUint::from_u32(61).unwrap();
let q = BigUint::from_u32(53).unwrap();
let rsa = RSA::new(p, q).expect("Error generating RSA");

// Encrypt and decrypt a numeric message
let message = BigUint::from_u32(42).unwrap();
let ciphertext = rsa.encrypt(message.clone()).unwrap();
let decrypted = rsa.decrypt(ciphertext).unwrap();

assert_eq!(message, decrypted);
```

### Byte Data with Padding

```rust
// Encrypt and decrypt byte data using PKCS#1 v1.5 padding
let msg_bytes = b"Hello RSA with padding!";
let cipher_bytes = rsa.encrypt_bytes(msg_bytes).unwrap();
let plain_bytes = rsa.decrypt_bytes(&cipher_bytes).unwrap();

assert_eq!(msg_bytes.to_vec(), plain_bytes);
```

## Security Considerations

1. **Key Size**: In real-world applications, much larger primes should be used (typically 2048 bits or more).

2. **Random Number Generation**: Secure random number generation is crucial for key generation and padding.

3. **Padding**: PKCS#1 v1.5 padding is implemented to prevent attacks like:
   - Direct RSA (textbook RSA) is vulnerable to chosen-ciphertext attacks
   - Padding ensures that the same message encrypts to different ciphertexts each time

4. **Side-Channel Attacks**: This implementation does not include protections against timing attacks or other side-channel vulnerabilities.

## Performance Considerations

- RSA operations are computationally intensive, especially for large key sizes
- The modular exponentiation (`modpow`) is the most expensive operation
- For bulk data encryption, RSA is typically used to encrypt a symmetric key, which is then used with a faster algorithm like AES

---

**Note**: This implementation is for educational purposes. For production systems, use established cryptographic libraries that have undergone security audits.