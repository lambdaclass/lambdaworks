# FROST: Flexible Round-Optimized Schnorr Threshold Signatures

This example implements a 2-of-2 FROST threshold signature scheme based on [RFC 9591](https://www.rfc-editor.org/rfc/rfc9591.html). Two parties can jointly sign a message without either party learning the other's secret key share.

## Disclaimer

This implementation is not cryptographically secure due to non-constant time operations, so it must not be used in production. It is intended to be just an educational example.

## Background: Schnorr Signatures

A standard Schnorr signature works as follows:

- **Private key:** $s \in \mathbb{F}_p$
- **Public key:** $Y = sG$
- **Signing:** Choose random nonce $k$, compute $R = kG$, challenge $c = H(R \| Y \| m)$, and signature $z = k + sc$
- **Verification:** Check $zG = R + cY$

FROST distributes the private key $s$ among multiple parties so that no single party knows $s$, yet they can collaboratively produce a valid Schnorr signature.

## Protocol Overview

### 1. Key Generation (Shamir Secret Sharing)

We use a degree-1 polynomial to split the secret:

$$f(x) = s + ax$$

where $s$ is the secret and $a$ is a random coefficient. Each party $i$ receives share $s_i = f(i)$:

- Party 1: $s_1 = f(1) = s + a$
- Party 2: $s_2 = f(2) = s + 2a$

The secret $s = f(0)$ can be reconstructed using **Lagrange interpolation**. For points at $x = 1$ and $x = 2$, the Lagrange coefficients at $x = 0$ are:

$$\lambda_1 = \frac{0 - 2}{1 - 2} = 2, \quad \lambda_2 = \frac{0 - 1}{2 - 1} = -1$$

Verification: $\lambda_1 s_1 + \lambda_2 s_2 = 2(s + a) + (-1)(s + 2a) = 2s + 2a - s - 2a = s$ ✓

The **group public key** is $Y = sG$, which everyone can compute but the secret $s$ is never reconstructed.

### 2. Signing Round 1: Nonce Commitments

Each party generates **two** random nonces (this is a key security feature of FROST):

- **Hiding nonce** $d_i$: Used directly in the combined nonce
- **Binding nonce** $e_i$: Scaled by a binding factor to prevent manipulation

Each party computes and broadcasts their commitments:

$$D_i = d_i G, \quad E_i = e_i G$$

### 3. Signing Round 2: Partial Signatures

After receiving all commitments, each party computes:

**Step 1: Binding factors** (prevent nonce manipulation)

$$\rho_i = H(\text{``rho''} \| Y \| \text{all commitments} \| \text{message} \| i)$$

The binding factor includes the participant's identifier, so each party has a unique $\rho_i$.

**Step 2: Combined nonce point**

$$R = \sum_{i} (D_i + \rho_i E_i)$$

**Step 3: Challenge**

$$c = H(\text{``chal''} \| R \| Y \| \text{message})$$

**Step 4: Partial signature**

$$z_i = d_i + e_i \rho_i + \lambda_i s_i c$$

### 4. Signature Aggregation

The final signature is $(R, z)$ where:

$$z = z_1 + z_2$$

### 5. Verification

The verifier checks:

$$zG \stackrel{?}{=} R + cY$$

**Why this works:**

$$z = z_1 + z_2 = (d_1 + e_1\rho_1 + \lambda_1 s_1 c) + (d_2 + e_2\rho_2 + \lambda_2 s_2 c)$$

$$= (d_1 + e_1\rho_1) + (d_2 + e_2\rho_2) + (\lambda_1 s_1 + \lambda_2 s_2)c$$

$$= (d_1 + e_1\rho_1) + (d_2 + e_2\rho_2) + sc$$

Therefore:

$$zG = (d_1 + e_1\rho_1)G + (d_2 + e_2\rho_2)G + scG$$

$$= (D_1 + \rho_1 E_1) + (D_2 + \rho_2 E_2) + cY = R + cY$$ ✓

## Why Two Nonces?

The binding factor $\rho_i$ depends on all commitments. This prevents a malicious party from:

1. Waiting to see other parties' commitments
2. Choosing their nonce to manipulate the final $R$

With a single nonce, a malicious party could compute their nonce as $k_{\text{bad}} = k_{\text{target}} - k_{\text{honest}}$ to force a specific $R$. The binding factor makes each party's effective nonce $d_i + \rho_i e_i$ unpredictable until after commitments are fixed.

## Implementation

| File | Description |
|------|-------------|
| `common.rs` | BN254 curve and field type definitions |
| `frost.rs` | RFC 9591 compliant FROST protocol |
| `main.rs` | Interactive demonstration |

### Key Functions



```rust
// Generate Shamir secret shares for 2 parties
let (share1, share2) = keygen();

// Round 1: Each party generates nonces and commitments
let (nonces1, commitment1) = sign_round1(&share1);
let (nonces2, commitment2) = sign_round1(&share2);

// Exchange commitments
let all_commitments = vec![commitment1, commitment2];

// Round 2: Each party computes their partial signature
let partial1 = sign_round2(&share1, &nonces1, &all_commitments, message)?;
let partial2 = sign_round2(&share2, &nonces2, &all_commitments, message)?;

let public_shares = vec![
    PublicShare {
        identifier: share1.identifier,
        public_share: share1.public_share.clone(),
    },
    PublicShare {
        identifier: share2.identifier,
        public_share: share2.public_share.clone(),
    },
];

// Aggregate into final signature
let signature = aggregate_signature(
    &share1.group_public_key,
    &all_commitments,
    &[partial1, partial2],
    &public_shares,
    message,
)?;

// Anyone can verify with just the public key
assert!(verify_signature(&share1.group_public_key, &signature, message)?);
```

## Running the Example

```bash
cargo run -p frost-signature
```

## References

- [RFC 9591: FROST](https://www.rfc-editor.org/rfc/rfc9591.html)
- [FROST Paper (Komlo & Goldberg, 2020)](https://eprint.iacr.org/2020/852)
