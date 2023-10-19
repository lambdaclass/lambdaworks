use lambdaworks_crypto::commitments::traits::IsCommitmentScheme;
use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
use lambdaworks_math::cyclic_group::IsGroup;
use lambdaworks_math::field::traits::{IsFFTField, IsField, IsPrimeField};
use lambdaworks_math::traits::{ByteConversion, Serializable};

use crate::prover::Proof;
use crate::setup::VerifyingKey;

pub struct Verifier {}

impl Verifier {
    pub fn new() -> Self {
        Self {}
    }

    // todo! check if args should be extended (e.g. public_input)
    pub fn verify(&self, p: &Proof, vk: &VerifyingKey) -> bool {
        // todo!
        true
    }
}

#[cfg(test)]
mod tests {
    // todo!
}
