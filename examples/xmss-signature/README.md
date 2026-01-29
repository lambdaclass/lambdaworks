# XMSS Signature Scheme

An educational implementation of XMSS (eXtended Merkle Signature Scheme) as defined in [RFC 8391](https://datatracker.ietf.org/doc/html/rfc8391).

## Overview

XMSS is a **hash-based signature scheme** that provides **post-quantum security**. Unlike ECDSA or RSA, which rely on the difficulty of factoring or discrete logarithms, XMSS security relies only on the security of the underlying hash function.

This makes XMSS a strong candidate for applications requiring long-term security guarantees, as it remains secure even against quantum computer attacks.

## Background: Ethereum's LeanSig

This implementation is inspired by Ethereum's [LeanSig proposal](https://eprint.iacr.org/2025/055) for post-quantum consensus signatures. The proposal suggests using XMSS-style signatures as a replacement for BLS signatures in Ethereum's consensus layer, providing a path to post-quantum security.

Key resources:
- [RFC 8391: XMSS](https://datatracker.ietf.org/doc/html/rfc8391)
- [Hash-Based Multi-Signatures for Post-Quantum Ethereum](https://eprint.iacr.org/2025/055)
- [Lean Consensus Roadmap](https://leanroadmap.org/)

## How XMSS Works

### 1. WOTS+ (Winternitz One-Time Signature)

WOTS+ is the foundation of XMSS. It uses hash chains to create one-time signatures:

```
Secret Key: sk = [sk_0, sk_1, ..., sk_66]  (67 random values)

Public Key: pk_i = F^(w-1)(sk_i)           (chain each value w-1 times)

Signature:  sig_i = F^(m_i)(sk_i)          (chain m_i times, where m_i is message digit)

Verify:     pk_i' = F^(w-1-m_i)(sig_i)     (complete the chain, should equal pk_i)
```

The checksum ensures that an attacker cannot forge signatures by only moving forward in chains.

### 2. L-Tree

The L-tree compresses the 67-element WOTS+ public key into a single 32-byte value using binary tree hashing. This compressed value becomes a leaf in the XMSS Merkle tree.

### 3. XMSS Merkle Tree

```
                    root (public key)
                   /                  \
              node                    node
             /    \                  /    \
          node    node            node    node
          /  \    /  \            /  \    /  \
        L0   L1  L2   L3  ...  L1020 L1021 L1022 L1023
```

- **Leaves**: L-tree compressed WOTS+ public keys
- **Root**: XMSS public key
- **Height**: h=10 allows 2^10 = 1024 signatures

### 4. Signing and Verification

**Signing** (uses leaf at index `idx`):
1. Generate randomness `r` for this signature
2. Hash message: `M' = H_msg(r || root || idx || message)`
3. Create WOTS+ signature of `M'`
4. Include authentication path from leaf to root

**Verification**:
1. Recompute `M'` from message and signature randomness
2. Recover WOTS+ public key from signature
3. Compress with L-tree to get leaf value
4. Use auth path to compute root
5. Compare computed root with public key

## Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| n         | 32    | Hash output size (bytes) |
| w         | 16    | Winternitz parameter |
| h         | 10    | Tree height |
| len_1     | 64    | Message hash chains |
| len_2     | 3     | Checksum chains |
| len       | 67    | Total WOTS+ chains |

**Signature size**: ~2500 bytes
**Public key size**: 64 bytes
**Available signatures**: 1024

## Usage

```rust
use xmss_signature::{Xmss, Sha256Hasher};

// Create XMSS instance
let xmss = Xmss::new(Sha256Hasher::new());

// Generate key pair (use secure random in production!)
let mut seed = [0u8; 96];
rand::thread_rng().fill_bytes(&mut seed);
let (public_key, mut secret_key) = xmss.keygen(&seed);

// Sign a message
let message = b"Hello, post-quantum world!";
let signature = xmss.sign(message, &mut secret_key)?;

// Verify
assert!(xmss.verify(message, &signature, &public_key));

// Check remaining signatures
println!("Remaining signatures: {}", secret_key.remaining_signatures());
```

## File Structure

```
src/
├── lib.rs           # Module exports and documentation
├── params.rs        # XMSS parameters (n, w, h, len)
├── address.rs       # ADRS structure for domain separation
├── hash.rs          # XmssHasher trait + SHA-256 implementation
├── utils.rs         # base_w conversion, checksum computation
├── wots.rs          # WOTS+ one-time signatures
├── ltree.rs         # L-Tree compression
├── xmss_tree.rs     # XMSS Merkle tree building and auth paths
└── xmss.rs          # Main API: keygen, sign, verify
```

## Security Warning

**This is an educational implementation.** It has not been audited for production use.

Critical considerations:

1. **Stateful Signatures**: XMSS requires careful state management. Each WOTS+ key pair can only be used **once**. Reusing a key pair allows signature forgery. The secret key index must be persisted reliably across system failures.

2. **No Side-Channel Protection**: This implementation does not include constant-time operations or other protections against timing attacks.

3. **Randomness Quality**: Key generation requires cryptographically secure random numbers.

4. **Key Exhaustion**: With h=10, only 1024 signatures are available per key pair. Monitor `remaining_signatures()` and generate new keys before exhaustion.

## Building and Testing

```bash
# Build
cargo build -p xmss-signature

# Run tests
cargo test -p xmss-signature

# Run tests with output
cargo test -p xmss-signature -- --nocapture
```

## License

Apache-2.0
