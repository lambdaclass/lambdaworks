# Schnorr Signature Scheme

The Schnorr signature scheme is a simple and efficient digital signature algorithm, whose security is based on the intractability of cthe discrete logarithm problem.  This example demonstrates an implementation using elliptic curves.

## ⚠️ Disclaimer
This implementation is not cryptographically secure due to non-constant time operations, so must not be used in production. It is intended to be just an educational example.

## What is a digital signature?

A digital signature is a cryptographic mechanism for signing messages that provides three fundamental properties:

- *Authentication:* Proves the identity of the signer.
- *Integrity:* Ensures the message has not been changed.
- *Non-repudiation:* The signer cannot deny having signed the message.

For example, let's say Alice wants to send Bob a message, and Bob wants to be sure that this message came from Alice. First Alice chooses a private and a public key. Then she sends Bob the message appending a signature which is computed from the message and her private key. Finallly Bob uses Alice's public key to verify the authenticity of the signed message.

## Schnorr Protocol

### Parameters known by the Signer and Verifier
- A group $G$ of prime order $p$ with generator $g$.
- A hash function $H$.

### Signer
- Choose a private key $k \in \mathbb{F}_p$.
- Compute the public key $h = g^{-k}$.
- Sample a random $ \ell \in \mathbb{F}_p$. This element can't be reused and must be sampled every time a new message 
- Compute $r = g^\ell \in G$
- Compute $e = H(r || M) \in \mathbb{F}_p$, where $M$ is the message.
- Compute $s = \ell + k \cdot e \in \mathbb{F}_p$.
- Sends $M$ with the signatur $(s, e)$.

### Verifier
- Compute $r_v = g^s \cdot h^e \in G$.
- Compute $e_v = H(r_v || M) \in \mathbb{F}_p$.
- Check $e_v = e$.

## Implementation
In `common.rs` you'll find the elliptic curve that we chose to use as the group. It can be used other elliptics curves or even other types of groups.

In `schnorr_signature.rs` you'll find the protocol. To test it see the below example:

```rust
// Choose a private key.
let private_key = FE::from(5);

// Compute the public key.
let public_key = SchnorrProtocol::get_public_key(&private_key);

// Write the message
let message = "hello world";

// Sign the message
let message_signed = SchnorrProtocol::sign_message(&private_key, message);

// Verify the signature
let is_the_signature_correct = SchnorrProtocol::verify_signature(
    &public_key,
    &message_signed
);
```




