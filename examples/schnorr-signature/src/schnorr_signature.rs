use crate::common::*;

use lambdaworks_math::{
    cyclic_group::IsGroup, elliptic_curve::traits::IsEllipticCurve, traits::ByteConversion,
};

use sha3::{Digest, Keccak256};

use rand::SeedableRng;

/// Schnorr Signature of a message using an elliptic curve as the group.
pub struct MessageSigned {
    pub message: String,
    pub signature: (FE, FE),
}

/// Function that should use the the signer to obtain her public key.
pub fn get_public_key(private_key: &FE) -> CurvePoint {
    let g = Curve::generator();
    // h = g^{-k}, where h = public_key and k = private_key.
    g.operate_with_self(private_key.canonical()).neg()
}

/// Function that should use the signer to sign a message.
pub fn sign_message(private_key: &FE, message: &str) -> MessageSigned {
    let g = Curve::generator();

    // Choose l a random field element. This element should be different in each signature.
    let rand = sample_field_elem(rand_chacha::ChaCha20Rng::from_entropy());

    // r = g^l.
    let r = g.operate_with_self(rand.canonical());

    // We want to compute e = H(r || message).
    let mut hasher = Keccak256::new();

    // We append r to the hasher.
    let r_coordinate_x_bytes = &r.to_affine().x().to_bytes_be();
    let r_coordinate_y_bytes = &r.to_affine().y().to_bytes_be();
    hasher.update(r_coordinate_x_bytes);
    hasher.update(r_coordinate_y_bytes);

    // We append the message to the hasher.
    let message_bytes = message.as_bytes();
    hasher.update(message_bytes);

    // e = H(r || message)
    let hashed_data = hasher.finalize().to_vec();
    let e = FE::from_bytes_be(&hashed_data).unwrap();

    // s = l + private_key * e
    let s = rand + &(private_key * &e);

    MessageSigned {
        message: message.to_string(),
        signature: (s, e),
    }
}

/// Function that should use the verifier to verify the message signed sent by the signer.
pub fn verify_signature(public_key: &CurvePoint, message_signed: &MessageSigned) -> bool {
    let g = Curve::generator();
    let message = &message_signed.message;
    let (s, e) = &message_signed.signature;

    // rv = g^s * h^e, with h = public_key and (s, e) = signature.
    let rv = g
        .operate_with_self(s.canonical())
        .operate_with(&public_key.operate_with_self(e.canonical()));

    // We want to compute ev = H(rv || M).
    let mut hasher = Keccak256::new();

    // We append rv to the hasher.
    let rv_coordinate_x_bytes = &rv.to_affine().x().to_bytes_be();
    let rv_coordinate_y_bytes = &rv.to_affine().y().to_bytes_be();
    hasher.update(rv_coordinate_x_bytes);
    hasher.update(rv_coordinate_y_bytes);

    // We append the message to the hasher.
    let message_bytes = message.as_bytes();
    hasher.update(message_bytes);

    // ev = H(rv || M).
    let hashed_data = hasher.finalize().to_vec();
    let ev = FE::from_bytes_be(&hashed_data).unwrap();

    // Check if H(rv || M) = H(r || M)
    ev == *e
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_valid_signature() {
        // The signer chooses a private key.
        let private_key = FE::from(5);
        // The signer calculates the public key that will be known by the verifier.
        let public_key = get_public_key(&private_key);
        // The signer signs the message.
        let message = "hello world";
        let message_signed = sign_message(&private_key, message);
        // The verifier verifies the message signed.
        assert!(verify_signature(&public_key, &message_signed));
    }

    #[test]
    // For security reasons, if the signer signs the same message twice (with the same private key),
    // the signatures should be different each time. This difference is achieved by the random field
    // element sampled for every message signed.
    fn same_message_signed_twice_has_different_signatures_each_time() {
        let private_key =
            FE::from_hex_unchecked("73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff");
        let public_key = get_public_key(&private_key);
        let message = "5f103b0bd4397d4df560eb559f38353f80eeb6";

        let message_signed_1 = sign_message(&private_key, message);
        let signatue_1 = &message_signed_1.signature;
        assert!(verify_signature(&public_key, &message_signed_1));

        let message_signed_2 = sign_message(&private_key, message);
        let signatue_2 = &message_signed_2.signature;
        assert!(verify_signature(&public_key, &message_signed_2));

        // The message is the same, but the signatures are different.
        assert_ne!(signatue_1, signatue_2);
    }
}
