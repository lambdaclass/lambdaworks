# RSA Implementation

This is an implementation of the RSA cryptographic algorithm in Rust. RSA is one of the first public-key cryptosystems widely used for secure data transmission.


## ⚠️ Disclaimer

This implementation is not cryptographically secure due to non-constant time operations and other considerations, so it must not be used in production. It is intended to be just an educational example.

## Overview

RSA is an asymmetric cryptographic algorithm that uses a pair of keys:
- **Public key**: Used for encryption and is shared openly
- **Private key**: Used for decryption and must be kept secret

The security of RSA relies on the practical difficulty of factoring the product of two large prime numbers.

## Mathematical Background

### Key Generation

1. Choose two distinct prime numbers $p$ and $q$
2. Compute $n = p \cdot q$
3. Calculate Euler's totient function: $\phi(n) = (p-1) \cdot (q-1)$
4. Choose an integer $e$ such that $1 < e < \phi(n)$ and $\gcd(e, \phi(n)) = 1$
5. Compute $d$ such that $d \cdot e \equiv 1 \pmod{\phi(n)}$

The public key is $(e, n)$ and the private key is $d$.

### Encryption and Decryption

- **Encryption**: $c = m^e \pmod{n}$ where $m$ is the message and $c$ is the ciphertext
- **Decryption**: $m = c^d \pmod{n}$

### PKCS#1 v1.5 Padding

For secure encryption of arbitrary byte data, we implement PKCS#1 v1.5 padding:

```
00 || 02 || PS || 00 || M
```

Where:
- `00`: First byte (block type)
- `02`: Second byte (block type for encryption)
- `PS`: Padding string of non-zero random bytes
- `00`: Separator
- `M`: Original message

### Basic  Example

```rust
use rsa::RSA;
use lambdaworks_math::unsigned_integer::element::UnsignedInteger;

// Create an RSA instance with primes that ensure e < φ(n)
let p = UnsignedInteger::<16>::from_u64(65539);
let q = UnsignedInteger::<16>::from_u64(65521);
let rsa = RSA::<16>::new(p, q)?;

// Encrypt and decrypt a numeric message
let message = UnsignedInteger::<16>::from_u64(42);
let ciphertext = rsa.encrypt(&message)?;
let decrypted = rsa.decrypt(&ciphertext)?;

assert_eq!(message, decrypted);
```

### Byte Data with Padding

```rust
use rsa::RSA;
use lambdaworks_math::unsigned_integer::element::UnsignedInteger;

// Create an RSA instance with primes that ensure e < φ(n)
let p = UnsignedInteger::<16>::from_u64(65539);
let q = UnsignedInteger::<16>::from_u64(65521);
let rsa = RSA::<16>::new(p, q)?;

// Encrypt and decrypt byte data using PKCS#1 v1.5 padding
let msg_bytes = b"Hello RSA with padding!";
let cipher_bytes = rsa.encrypt_bytes_pkcs1(msg_bytes)?;
let plain_bytes = rsa.decrypt_bytes_pkcs1(&cipher_bytes)?;

assert_eq!(msg_bytes.to_vec(), plain_bytes);
```

---

**Note**: This implementation is for educational purposes. Production systems should use established cryptographic libraries that have undergone security audits.
