use lambdaworks_math::field::{
    element::FieldElement, fields::fft_friendly::babybear::Babybear31PrimeField,
};

/*
A polynomial is a vector
if the polynomial is a_{n-1}* x^{n-1}+ ...+a_0
the vector is [a_0, a_1,...., a_{n-1}]
where i=i_1...i_n, that is i_j are the digits of i in base 2, and x^i= x_1^{i_1}.... x_n^{i_n}
For n variables, this vector will be of length 2^n
*/
pub type poly = Vec<FieldElement<Babybear31PrimeField>>;

pub enum ProverMessage {
    Sum(FieldElement<Babybear31PrimeField>),
    Polynomial(poly),
}

pub fn sumcheck_prover(round: u64, p: poly) -> ProverMessage {
    if round == 0 {
        ProverMessage::Sum(FieldElement::<Babybear31PrimeField>::one())
    } else if round == 1 {
        ProverMessage::Sum(FieldElement::<Babybear31PrimeField>::one())
    } else {
        ProverMessage::Sum(FieldElement::<Babybear31PrimeField>::one())
    }
}

fn eval_polynomial(var_asignment: u64, p: poly) {
    
    for var in 0..p.len() {
        
    }
}

#[cfg(test)]
mod test_prover {

    #[test]
    fn aaa() {}
}
