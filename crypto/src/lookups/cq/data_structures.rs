use lambdaworks_math::{
    polynomial::Polynomial,
    field::{traits::{IsField, IsFFTField}, element::{self, FieldElement}},
    elliptic_curve::traits::IsPairing,
};

// srs key used to commit statment to table and in final cq lookup
pub struct ProvingKey<F: IsPairing> {
    pub(crate) srs_g1: F::G1Point,
}

// Statement we are trying to prove within the Lookup Table
pub struct Statement<F: IsFFTField + IsField> {
    pub(crate) size: usize,
    pub(crate) f: Polynomial<FieldElement<F>>,
    pub(crate) f_evals: Vec<FieldElement<F>>,
}

impl<F: IsFFTField + IsField> Statement<F> {
    //TODO: Figure out iFFT
    pub fn new(witnesses: &Vec<FieldElement<F>>) -> Self {
        if !witnesses.len().is_power_of_two() {
            panic!("witnes size not power of 2")
        }

        let f = Polynomial { coefficients: witnesses.clone() };
       Self {
        size: witnesses.len(),
        f,
        f_evals: witnesses.clone(),
       }
    }
}

pub struct Proof<F: IsPairing> {
    pub(crate) first_msg: ProverFirstMsg<F>,
    pub(crate) second_msg: ProverSecondMsg<F>,
    pub(crate) third_msg: ProverThirdMsg<F>,
}

pub struct ProverFirstMsg<F: IsPairing> {
    pub(crate) m_cm: F::G1Point,
}

pub struct ProverSecondMsg<F: IsPairing> {
    pub(crate) a_cm: F::G1Point,
    pub(crate) q_cm: F::G1Point,
    pub(crate) b0_cm: F::G1Point,
    pub(crate) qb_cm: F::G1Point,
    pub(crate) p_cm: F::G1Point,
}

pub struct ProverThirdMsg<F: IsPairing> {
    //TODO: check if this is the right field otherwise extend IsPairing trait
    pub(crate) b0_at_gamma: F::OutputField,
    pub(crate) f_at_gamma: F::OutputField,
    pub(crate) a_at_zero: F::OutputField,
    pub(crate) pi_gamma: F::G1Point,
    pub(crate) a0_cm: F::G1Point
}