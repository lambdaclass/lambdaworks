use lambdaworks_crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsField, IsSubFieldOf},
    },
    traits::AsBytes,
};

use crate::{
    proof::stark::StarkProof,
    prover::{IsStarkProver, Prover, ProvingError},
    trace::TraceTable,
    traits::AIR,
};

// List of airs and their associated table
type Airs<'a, F, E, PI> = Vec<(
    Box<dyn AIR<Field = F, FieldExtension = E, PublicInputs = PI>>,
    &'a mut TraceTable<F, E>,
)>;
pub fn multi_prove<
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: Send + Sync + IsFFTField,
    PI: Send + Sync,
>(
    airs: Airs<F, E, PI>,
    transcript: &mut impl IsStarkTranscript<E, F>,
) -> Result<StarkProof<F, E>, ProvingError>
where
    FieldElement<F>: AsBytes + Send + Sync,
    FieldElement<E>: AsBytes + Send + Sync,
    <F as IsField>::BaseType: Sync + Send,
    <E as IsField>::BaseType: Sync + Send,
{
    let mut proof = None;
    for (air, table) in airs {
        let _ = proof.insert(Prover::<F, E, PI>::prove(&air, table, transcript)?);
    }
    Ok(proof.unwrap())
}
