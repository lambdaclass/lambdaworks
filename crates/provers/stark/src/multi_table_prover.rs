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
    proof::{options::ProofOptions, stark::StarkProof},
    prover::{IsStarkProver, Prover, ProvingError},
    trace::TraceTable,
    traits::AIR,
};

fn multi_prove<F: IsSubFieldOf<E> + IsFFTField + Send + Sync, E: Send + Sync + IsFFTField>(
    airs: Vec<(
        Box<dyn AIR<Field = F, FieldExtension = E, PublicInputs = Vec<F>>>,
        &mut TraceTable<F, E>,
    )>,
    pub_inputs: &Vec<F>,
    proof_options: &ProofOptions,
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
        proof.insert(Prover::<F, E>::prove(
            air.as_ref(),
            table,
            pub_inputs,
            proof_options,
            transcript,
        )?);
    }
    Ok(proof.unwrap())
}
