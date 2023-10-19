use lambdaworks_crypto::commitments::traits::IsCommitmentScheme;
use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
use lambdaworks_math::fft::polynomial::FFTPoly;
use lambdaworks_math::field::traits::IsFFTField;
use lambdaworks_math::field::{element::FieldElement, traits::IsField};
use lambdaworks_math::traits::{ByteConversion, Serializable};

pub struct ProvingKey {
    // NbG1 returns the number of G1 elements in the ProvingKey
    pub nb_g1: u8,
    // NbG2 returns the number of G2 elements in the ProvingKey
    pub nb_g2: u8,
    pub is_different: bool,
}

pub struct VerifyingKey {
    // number of elements expected in the public witness
    pub NbPublicWitness: u8,

    // number of G1 elements in the VerifyingKey
    pub NbG1: u8,

    // number of G2 elements in the VerifyingKey
    pub NbG2: u8,

    pub is_different: bool,
}

pub struct KeyWrapper {
    pub verifying_key: VerifyingKey,
    pub proving_key: ProvingKey,
}

pub struct Witness<F: IsField> {
    pub a: Vec<FieldElement<F>>,
    pub b: Vec<FieldElement<F>>,
    pub c: Vec<FieldElement<F>>,
}

pub fn setup() -> KeyWrapper {
    KeyWrapper {
        verifying_key: VerifyingKey {
            NbPublicWitness: 0,
            NbG1: 0,
            NbG2: 0,
            is_different: false,
        },
        proving_key: ProvingKey {
            nb_g1: 0,
            nb_g2: 0,
            is_different: false,
        },
    }
}

#[cfg(test)]
mod tests {
    // todo!
}
