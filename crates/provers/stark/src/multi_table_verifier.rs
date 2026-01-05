use lambdaworks_crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsSubFieldOf},
    },
    traits::AsBytes,
};

use crate::{
    proof::stark::StarkProof,
    traits::AIR,
    verifier::{IsStarkVerifier, Verifier},
};

// List of airs and their associated proof
type AirsAndProofs<'a, F, E, PI> = Vec<(
    &'a dyn AIR<Field = F, FieldExtension = E, PublicInputs = PI>,
    &'a StarkProof<F, E>,
)>;

/// Verifies multiple STARK proofs with their corresponding airs `airs_and_proofs`.
/// Warning: the transcript must be safely initializated before passing it to this method.
pub fn multi_verify<
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: Send + Sync + IsFFTField,
    PI,
>(
    airs_and_proofs: AirsAndProofs<F, E, PI>,
    transcript: &mut impl IsStarkTranscript<E, F>,
) -> bool
where
    FieldElement<F>: AsBytes,
    FieldElement<E>: AsBytes,
{
    for (air, proof) in airs_and_proofs {
        if !Verifier::verify(proof, air, transcript) {
            return false;
        }
    }
    true
}
