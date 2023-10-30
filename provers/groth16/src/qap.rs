use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrElement;
use lambdaworks_math::polynomial::Polynomial;

#[derive(Debug)]
pub struct QAP {
    pub num_of_public_inputs: usize,
    pub num_of_total_inputs: usize,
    pub num_of_gates: usize,
    pub l: Vec<Polynomial<FrElement>>,
    pub r: Vec<Polynomial<FrElement>>,
    pub o: Vec<Polynomial<FrElement>>,
    pub t: Polynomial<FrElement>,
}

impl QAP {
    pub fn new(
        num_of_public_inputs: usize,
        gate_indices: Vec<FrElement>,
        l: Vec<Vec<&str>>,
        r: Vec<Vec<&str>>,
        o: Vec<Vec<&str>>,
    ) -> Self {
        assert!(l.len() == r.len() && r.len() == o.len() && num_of_public_inputs <= l.len());

        let mut target_poly = Polynomial::new(&[FrElement::one()]);
        for gate_index in &gate_indices {
            target_poly = target_poly * Polynomial::new(&[-gate_index, FrElement::one()]);
        }

        Self {
            num_of_public_inputs,
            num_of_total_inputs: l.len(),
            num_of_gates: gate_indices.len(),
            l: Self::build_test_variable_polynomial(&gate_indices, &l),
            r: Self::build_test_variable_polynomial(&gate_indices, &r),
            o: Self::build_test_variable_polynomial(&gate_indices, &o),
            t: target_poly,
        }
    }

    fn build_test_variable_polynomial(
        gate_indices: &Vec<FrElement>,
        from_matrix: &Vec<Vec<&str>>,
    ) -> Vec<Polynomial<FrElement>> {
        let mut polynomials = vec![];
        for i in 0..from_matrix.len() {
            let mut y_indices = vec![];
            for string in &from_matrix[i] {
                y_indices.push(FrElement::from_hex_unchecked(*string));
            }
            polynomials.push(Polynomial::interpolate(gate_indices, &y_indices).unwrap());
        }
        polynomials
    }
}
