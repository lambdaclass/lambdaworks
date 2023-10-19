use lambdaworks_crypto::commitments::traits::IsCommitmentScheme;
use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
use lambdaworks_math::{field::traits::IsField, traits::ByteConversion};
use lambdaworks_math::fft::polynomial::FFTPoly;
use lambdaworks_math::field::traits::IsFFTField;
use lambdaworks_math::traits::{Deserializable, IsRandomFieldElementGenerator, Serializable};

use crate::setup::ProvingKey;

pub struct Proof {
    //todo!
}

pub fn prove(r1cs_constraint_system: u8, pk: ProvingKey, witness: u8) -> Result<Proof, String> {
    // todo! - change r1cs_constraint_system and witness data types
    // todo! - implement method
    Result::Ok(Proof {})
}

// todo - implement Serializable/Deserializable traits for Proofs

// todo - Should we define a common trait to be used by all provers (Groth16, Stark, Plonk)?
pub struct Groth16Prover {
    // todo!
}

impl Groth16Prover
{
    fn new() -> Self {
        Self {}
    }

    fn prove(&self) -> Result<Proof, String> {
        //todo!
        Result::Ok(Proof {})
    }
}

#[cfg(test)]
mod tests {
    // todo! implement tests
}
