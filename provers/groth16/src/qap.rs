use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrElement;
use lambdaworks_math::field::element::FieldElement;
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
            target_poly = target_poly * Polynomial::new(&[-gate_index, FieldElement::one()]);
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

    /// Builds QAP representation for equation x^3 + x + 5 = 35, based on Vitalik's post
    ///https://vitalik.ca/general/2016/12/10/qap.html
    pub fn build_example() -> QAP {
        QAP::new(
            1,
            // TODO: Roots of unity
            ["0x1", "0x2", "0x3", "0x4"]
                .map(|e| FrElement::from_hex_unchecked(e))
                .to_vec(),
            [
                ["0", "0", "0", "5"],
                ["1", "0", "1", "0"],
                ["0", "0", "0", "0"],
                ["0", "1", "0", "0"],
                ["0", "0", "1", "0"],
                ["0", "0", "0", "1"],
            ]
                .map(|col| col.to_vec())
                .to_vec(),
            [
                ["0", "0", "1", "1"],
                ["1", "1", "0", "0"],
                ["0", "0", "0", "0"],
                ["0", "0", "0", "0"],
                ["0", "0", "0", "0"],
                ["0", "0", "0", "0"],
            ]
                .map(|col| col.to_vec())
                .to_vec(),
            [
                ["0", "0", "0", "0"],
                ["0", "0", "0", "0"],
                ["0", "0", "0", "1"],
                ["1", "0", "0", "0"],
                ["0", "1", "0", "0"],
                ["0", "0", "1", "0"],
            ]
                .map(|col| col.to_vec())
                .to_vec(),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::qap::QAP;

    #[test]
    fn test_example() {
        let qap = QAP::build_example();
        assert_eq!(qap.num_of_gates, 4);
    }
}