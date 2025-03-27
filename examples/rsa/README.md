# RSA Encryption Example

This is a simple implementation of the RSA encryption algorithm using the `lambdaworks_math` library. The implementation demonstrates the core concepts of RSA encryption with small numbers for educational purposes.

## Mathematical Background

RSA is an asymmetric cryptographic algorithm that uses a pair of keys:
- **Public key**: Used for encryption and is shared openly
- **Private key**: Used for decryption and must be kept secret

The security of RSA relies on the practical difficulty of factoring the product of two large prime numbers.

### Key Generation

1. Choose two distinct prime numbers $p$ and $q$
2. Compute $n = p \cdot q$
3. Calculate Euler's totient function: $\phi(n) = (p-1) \cdot (q-1)$
4. Choose an integer $e$ such that $1 < e < \phi(n)$ and $\gcd(e, \phi(n)) = 1$
5. Compute $d$ such that $d \cdot e \equiv 1 \pmod{\phi(n)}$

The public key is $(e, n)$ and the private key is $d$.

### Euler's Totient Function

Euler's totient function $\phi(n)$ counts the positive integers up to $n$ that are coprime to $n$. For a prime number $p$, $\phi(p) = p-1$ since all numbers less than $p$ are coprime to $p$.

For a product of two primes $n = p \cdot q$, $\phi(n) = \phi(p) \cdot \phi(q) = (p-1) \cdot (q-1)$.

### Encryption and Decryption

- **Encryption**: $c = m^e \pmod{n}$ where $m$ is the message and $c$ is the ciphertext
- **Decryption**: $m = c^d \pmod{n}$


## Implementation Details

The implementation uses:
- 4 limbs for the `UnsignedInteger` type
- Public exponent `e = 65537` (0x10001)
- Small prime numbers for demonstration (not secure for real use)
- No padding scheme (left as an exercise for the reader)

### Key Components

1. `RSA` struct:
   - `e`: Public exponent (65537)
   - `d`: Private exponent
   - `n`: Modulus (product of two primes)

2. Main functions:
   - `new(p, q)`: Creates a new RSA instance from two primes
   - `encrypt(message)`: Encrypts a numeric message
   - `decrypt(ciphertext)`: Decrypts a numeric ciphertext
   - `encrypt_bytes_simple(msg)`: Encrypts a byte array
   - `decrypt_bytes_simple(cipher)`: Decrypts a byte array

3. Helper functions:
   - `modinv(a, m)`: Computes modular inverse using extended Euclidean algorithm
   - `modpow(base, exponent, modulus)`: Computes modular exponentiation using square-and-multiply

## Example Usage

```rust
// Create RSA instance with primes p=61 and q=53
let p = UnsignedInteger::from_u64(61);
let q = UnsignedInteger::from_u64(53);
let rsa = RSA::new(p, q).unwrap();

// Encrypt a message
let message = UnsignedInteger::from_u64(42);
let ciphertext = rsa.encrypt(&message).unwrap();

// Decrypt the ciphertext
let decrypted = rsa.decrypt(&ciphertext).unwrap();
assert_eq!(message, decrypted);
```

## Running the Examples

```bash
# Run the examples
cargo run

# Run the tests
cargo test
```
