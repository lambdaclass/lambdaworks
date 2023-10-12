use std::marker::PhantomData;

use lambdaworks_math::{
    field::{element::FieldElement, traits::IsFFTField},
    traits::Serializable,
};

use crate::{prover::IsStarkProver, verifier::IsStarkVerifier};



#[cfg(test)]
pub mod tests {
    use std::num::ParseIntError;

    use lambdaworks_math::field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    };

    use crate::{
        domain::Domain,
        examples::fibonacci_2_cols_shifted::{self, Fibonacci2ColsShifted},
        proof::options::ProofOptions,
        prover::{IsStarkProver, Prover},
        traits::AIR,
        transcript::StoneProverTranscript,
        verifier::{IsStarkVerifier, Verifier},
    };

}
