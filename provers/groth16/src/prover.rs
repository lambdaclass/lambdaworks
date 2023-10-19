use crate::setup::ProvingKey;

pub struct Proof {
    //todo!
}

// todo - implement Serializable/Deserializable traits for Proofs

// todo - Should we define a common trait to be used by all provers (Groth16, Stark, Plonk)?

#[derive(Default)]
pub struct Groth16Prover {
    // todo!
}

impl Groth16Prover {
    pub fn prove(
        &self,
        _r1cs_constraint_system: u8,
        _pk: ProvingKey,
        _witness: u8,
    ) -> Result<Proof, String> {
        // todo! - change r1cs_constraint_system and witness data types
        // todo! - implement method
        Ok(Proof {})
    }
}

#[cfg(test)]
mod tests {
    // todo! implement tests
}
