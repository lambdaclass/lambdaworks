use cairo_platinum_prover::air::CairoAIR;
use stark_platinum_prover::proof::options::ProofOptions;
use stark_platinum_prover::proof::options::SecurityLevel;
use stark_platinum_prover::transcript::StoneProverTranscript;
use stark_platinum_prover::verifier::{IsStarkVerifier, Verifier};

fn verify_cairo_proof_ffi(proof_bytes: &[u8], proof_options: &ProofOptions) -> bool {
    let bytes = proof_bytes;

    // This logic is the same as main verify, with only error handling changing. In ffi, we simply return a false if the proof is invalid, instead of rising an error.

    // Proof len was stored as an u32, 4u8 needs to be read
    let proof_len = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;

    let bytes = &bytes[4..];
    if bytes.len() < proof_len {
        return false;
    }

    let Ok((proof, _)) =
        bincode::serde::decode_from_slice(&bytes[0..proof_len], bincode::config::standard())
    else {
        return false;
    };
    let bytes = &bytes[proof_len..];

    let Ok((pub_inputs, _)) = bincode::serde::decode_from_slice(bytes, bincode::config::standard())
    else {
        return false;
    };

    Verifier::<CairoAIR>::verify(
        &proof,
        &pub_inputs,
        proof_options,
        StoneProverTranscript::new(&[]),
    )
}

// Fibo 70k is 260 kb
// 2 MiB is more than enough
const MAX_PROOF_SIZE: usize = 1024 * 1024;

/// WASM Function for verifying a proof with default 100 bits of security
#[no_mangle]
pub extern "C" fn verify_cairo_proof_ffi_100_bits(
    proof_bytes: &[u8; MAX_PROOF_SIZE],
    real_len: usize,
) -> bool {
    let (real_proof_bytes, _) = proof_bytes.split_at(real_len);

    println!("Len: {:?} ", real_proof_bytes.len());
    let proof_options = ProofOptions::new_secure(SecurityLevel::Conjecturable100Bits, 3);
    verify_cairo_proof_ffi(real_proof_bytes, &proof_options)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROOF: &[u8; 265809] = include_bytes!("../fibo_5.proof");

    #[test]
    fn fibo_5_proof_verifies() {
        let mut proof_buffer = [0u8; super::MAX_PROOF_SIZE];
        let proof_size = PROOF.len();
        proof_buffer[..proof_size].clone_from_slice(PROOF);
        let result = verify_cairo_proof_ffi_100_bits(&proof_buffer, proof_size);
        assert!(result)
    }
}
