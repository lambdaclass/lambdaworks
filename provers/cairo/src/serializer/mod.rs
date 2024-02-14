use lambdaworks_math::{
    field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField, traits::AsBytes,
};
use stark_platinum_prover::{
    proof::{options::ProofOptions, stark::StarkProof},
    traits::AIR,
};

use self::{
    stark_config::StarkConfig, stark_public_input::CairoPublicInput,
    stark_unsent_commitment::StarkUnsentCommitment, stark_witness::StarkWitness,
};

pub mod stark_config;
pub mod stark_public_input;
pub mod stark_unsent_commitment;
pub mod stark_witness;

pub struct CairoStarkProof {
    pub stark_config: StarkConfig,
    pub public_input: CairoPublicInput,
    pub unsent_commitment: StarkUnsentCommitment,
    pub witness: StarkWitness,
}

/// Serializer compatible with cairoX-verifier stark proof struct
pub struct CairoCompatibleSerializer;

impl CairoCompatibleSerializer {}

impl CairoCompatibleSerializer {
    pub fn convert_proof<A>(
        proof: &StarkProof<Stark252PrimeField, Stark252PrimeField>,
        public_inputs: &A::PublicInputs,
        options: &ProofOptions,
    ) -> CairoStarkProof
    where
        A: AIR<Field = Stark252PrimeField, FieldExtension = Stark252PrimeField>,
        A::PublicInputs: AsBytes,
    {
        todo!()
    }
}
