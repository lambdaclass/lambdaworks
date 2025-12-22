use lambdaworks_crypto::fiat_shamir::is_transcript::IsStarkTranscript;
use lambdaworks_math::{
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsField, IsSubFieldOf},
    },
    traits::AsBytes,
};

use crate::{
    domain::new_domain,
    proof::stark::StarkProof,
    prover::{IsStarkProver, Prover, ProvingError},
    trace::TraceTable,
    traits::AIR,
};

fn multi_prove<
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: Send + Sync + IsFFTField,
    PI: Send + Sync,
>(
    airs: Vec<(
        Box<dyn AIR<Field = F, FieldExtension = E, PublicInputs = PI>>,
        &mut TraceTable<F, E>,
    )>,
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
        let domain = new_domain(air.as_ref());
        proof.insert(Prover::<F, E, PI>::prove(air.as_ref(), table, transcript)?);
    }
    Ok(proof.unwrap())
}
