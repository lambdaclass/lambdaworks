use crate::{common::*, r1cs::R1CS};
use lambdaworks_math::polynomial::Polynomial;
use std::convert::From;

#[derive(Clone, Debug, PartialEq, Eq)]
/// QAP Representation of the circuits
pub struct QuadraticArithmeticProgram {
    pub vs: Vec<Polynomial<FE>>,
    pub ws: Vec<Polynomial<FE>>,
    pub ys: Vec<Polynomial<FE>>,
    pub target: Polynomial<FE>,
    pub number_of_inputs: usize,
    pub number_of_outputs: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub enum CreationError {
    PolynomialVectorsSizeMismatch,
}

impl QuadraticArithmeticProgram {
    /// Creates a new QAP
    /// This expects vectors to be organized like:
    /// v0,w0,y0
    /// inputs associated v,w,y polynomials
    /// mid associated polynomials
    /// outputs associated v,w,y polynomials
    pub fn new(
        vs: Vec<Polynomial<FE>>,
        ws: Vec<Polynomial<FE>>,
        ys: Vec<Polynomial<FE>>,
        target: Polynomial<FE>,
        number_of_inputs: usize,
        number_of_outputs: usize,
    ) -> Result<Self, CreationError> {
        if vs.len() != ws.len()
            || vs.len() != ys.len()
            || number_of_inputs + number_of_outputs > vs.len()
        {
            Err(CreationError::PolynomialVectorsSizeMismatch)
        } else {
            Ok(Self {
                vs,
                ws,
                ys,
                target,
                number_of_inputs,
                number_of_outputs,
            })
        }
    }

    pub fn h_polynomial(&self, c: &[FE]) -> Polynomial<FE> {
        self.p_polynomial(c).div_with_ref(&self.target)
    }

    /// Receives C elements of a solution of the circuit.
    /// Returns p polynomial.
    // This along the polynomial execution should be migrated with a better
    // representation of the circuit.
    pub fn p_polynomial(&self, cs: &[FE]) -> Polynomial<FE> {
        let v: Polynomial<FE> = self.vs[0].clone()
            + self.vs[1..]
                .iter()
                .zip(cs)
                .map(|(v, c)| v.mul_with_ref(&Polynomial::new_monomial(c.clone(), 0)))
                .reduce(|x, y| x + y)
                .unwrap();

        let w: Polynomial<FE> = self.ws[0].clone()
            + self.ws[1..]
                .iter()
                .zip(cs)
                .map(|(w, c)| w.mul_with_ref(&Polynomial::new_monomial(c.clone(), 0)))
                .reduce(|x, y| x + y)
                .unwrap();

        let y: Polynomial<FE> = self.ys[0].clone()
            + self.ys[1..]
                .iter()
                .zip(cs)
                .map(|(y, c)| y.mul_with_ref(&Polynomial::new_monomial(c.clone(), 0)))
                .reduce(|x, y| x + y)
                .unwrap();

        v * w - y
    }

    pub fn v_mid(&'_ self) -> &[Polynomial<FE>] {
        &self.vs[self.number_of_inputs + 1..(self.vs.len() - self.number_of_outputs)]
    }

    pub fn w_mid(&'_ self) -> &[Polynomial<FE>] {
        &self.ws[self.number_of_inputs + 1..(self.ws.len() - self.number_of_outputs)]
    }

    pub fn y_mid(&'_ self) -> &[Polynomial<FE>] {
        &self.ys[self.number_of_inputs + 1..(self.ys.len() - self.number_of_outputs)]
    }

    pub fn v_input(&'_ self) -> &[Polynomial<FE>] {
        &self.vs[1..self.number_of_inputs + 1]
    }

    pub fn w_input(&'_ self) -> &[Polynomial<FE>] {
        &self.ws[1..self.number_of_inputs + 1]
    }

    pub fn y_input(&'_ self) -> &[Polynomial<FE>] {
        &self.ys[1..self.number_of_inputs + 1]
    }

    pub fn v0(&'_ self) -> &Polynomial<FE> {
        &self.vs[0]
    }

    pub fn w0(&'_ self) -> &Polynomial<FE> {
        &self.ws[0]
    }

    pub fn y0(&'_ self) -> &Polynomial<FE> {
        &self.ys[0]
    }

    pub fn v_output(&'_ self) -> &[Polynomial<FE>] {
        &self.vs[(self.vs.len() - self.number_of_outputs)..]
    }
    pub fn w_output(&'_ self) -> &[Polynomial<FE>] {
        &self.ws[(self.ws.len() - self.number_of_outputs)..]
    }

    pub fn y_output(&'_ self) -> &[Polynomial<FE>] {
        &self.ys[(self.ys.len() - self.number_of_outputs)..]
    }
}

impl From<R1CS> for QuadraticArithmeticProgram {
    fn from(r1cs: R1CS) -> Self {
        // The r values for the qap polynomial can each be any number,
        // as long as there are the right amount of rs.
        // In this case, it's set them to be 0,1,2..number_of_constraints(),
        // number_of_constraints non inclusive.
        let rs: Vec<FE> = (0..r1cs.number_of_constraints() as u128)
            .map(|i| FE::new(i.into()))
            .collect();

        let mut vs: Vec<Polynomial<FE>> = Vec::with_capacity(r1cs.witness_size());
        let mut ws: Vec<Polynomial<FE>> = Vec::with_capacity(r1cs.witness_size());
        let mut ys: Vec<Polynomial<FE>> = Vec::with_capacity(r1cs.witness_size());
        let mut t: Polynomial<FE> = Polynomial::new_monomial(FE::from(1), 0);

        for r in &rs {
            t = t * Polynomial::new(&[-r, FE::from(1)]);
        }

        for i in 0..r1cs.witness_size() {
            let v_ys: Vec<FE> = r1cs.constraints.iter().map(|c| c.a[i].clone()).collect();
            let w_ys: Vec<FE> = r1cs.constraints.iter().map(|c| c.b[i].clone()).collect();
            let y_ys: Vec<FE> = r1cs.constraints.iter().map(|c| c.c[i].clone()).collect();

            vs.push(Polynomial::interpolate(&rs, &v_ys).expect("should interpolate"));
            ws.push(Polynomial::interpolate(&rs, &w_ys).expect("should interpolate"));
            ys.push(Polynomial::interpolate(&rs, &y_ys).expect("should interpolate"));
        }

        QuadraticArithmeticProgram {
            vs,
            ws,
            ys,
            target: t,
            number_of_inputs: r1cs.number_of_inputs,
            number_of_outputs: r1cs.number_of_outputs,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::{
        new_test_qap, new_test_r1cs, test_qap_r5, test_qap_r6, test_qap_solver,
    };

    use super::*;

    #[test]
    fn qap_with_different_amount_of_polynomials_should_error() {
        let v = &[
            Polynomial::new(&[FE::from(1), FE::from(2)]),
            Polynomial::new(&[FE::from(2), FE::from(3)]),
        ];

        let u = v.clone();
        let w = &[Polynomial::new(&[FE::from(1), FE::from(2)])];
        let t = Polynomial::new(&[FE::from(3)]);
        assert_eq!(
            Err(CreationError::PolynomialVectorsSizeMismatch),
            QuadraticArithmeticProgram::new(v.to_vec(), u.to_vec(), w.to_vec(), t, 2, 1)
        );
    }

    #[test]
    fn test_circuit_v_w_y_have_7_elements() {
        let test_circuit = new_test_qap();
        assert_eq!(test_circuit.vs.len(), 7);
        assert_eq!(test_circuit.ws.len(), 7);
        assert_eq!(test_circuit.ys.len(), 7);
    }

    //_mid polynomials of test circuit contains only one polynomial
    #[test]
    fn v_mid_test_circuit_on_r6_is_0() {
        let test_circuit = new_test_qap();
        let r6 = test_qap_r6();
        assert_eq!(test_circuit.y_mid()[0].evaluate(&r6), FE::from(0));
    }

    #[test]
    fn w_mid_test_circuit_has_one_element() {
        let test_circuit = new_test_qap();
        assert_eq!(test_circuit.v_mid().len(), 1);
    }

    #[test]
    fn w_mid_test_circuit_on_r5_is_0() {
        let test_circuit = new_test_qap();
        let r5 = test_qap_r5();
        assert_eq!(test_circuit.w_mid()[0].evaluate(&r5), FE::from(0));
    }

    #[test]
    fn w_mid_test_circuit_on_r6_is_1() {
        let test_circuit = new_test_qap();
        let r6 = test_qap_r6();
        assert_eq!(test_circuit.w_mid()[0].evaluate(&r6), FE::from(1));
    }

    #[test]
    fn y_mid_test_circuit_on_r5_is_1() {
        let test_circuit = new_test_qap();
        let r5 = test_qap_r5();
        assert_eq!(test_circuit.y_mid()[0].evaluate(&r5), FE::from(1));
    }

    #[test]
    fn y_mid_test_circuit_on_r6_is_0() {
        let test_circuit = new_test_qap();
        let r6 = test_qap_r6();
        assert_eq!(test_circuit.y_mid()[0].evaluate(&r6), FE::from(0));
    }

    #[test]
    fn v_input_test_circuit_has_length_4() {
        let test_circuit = new_test_qap();
        assert_eq!(test_circuit.v_input().len(), 4);
    }

    #[test]
    fn w_input_test_circuit_has_length_4() {
        let test_circuit = new_test_qap();
        assert_eq!(test_circuit.w_input().len(), 4);
    }
    #[test]
    fn y_input_test_circuit_has_length_4() {
        let test_circuit = new_test_qap();
        assert_eq!(test_circuit.y_input().len(), 4);
    }

    #[test]
    fn v_output_test_circuit_has_length_1() {
        let test_circuit = new_test_qap();
        assert_eq!(test_circuit.v_output().len(), 1);
    }

    #[test]
    fn w_output_test_circuit_has_length_1() {
        let test_circuit = new_test_qap();
        assert_eq!(test_circuit.w_output().len(), 1);
    }

    #[test]
    fn y_output_test_circuit_has_length_1() {
        let test_circuit = new_test_qap();
        assert_eq!(test_circuit.y_output().len(), 1);
    }

    #[test]
    /// This test runs multiple cases calculated in paper
    /// t polynomial is tested implicitly by calculating h = p / t
    fn test_polynomial_h_cases() {
        let test_circuit = new_test_qap();

        let inputs = [FE::from(1), FE::from(2), FE::from(3), FE::from(4)];

        let (c5, c6) = test_qap_solver(inputs.clone());

        let mut c_vector = inputs.to_vec();
        c_vector.append(&mut vec![c5, c6]);

        assert_eq!(
            test_circuit.h_polynomial(&c_vector),
            Polynomial::new_monomial(FE::from(0), 0)
        );

        let inputs = [FE::from(2), FE::from(2), FE::from(2), FE::from(2)];

        let (c5, c6) = test_qap_solver(inputs.clone());

        let mut c_vector = inputs.to_vec();
        c_vector.append(&mut vec![c5, c6]);

        assert_eq!(
            test_circuit.h_polynomial(&c_vector),
            Polynomial::new_monomial(FE::from(4), 0)
        );

        let inputs = [FE::from(3), FE::from(3), FE::from(3), FE::from(3)];

        let (c5, c6) = test_qap_solver(inputs.clone());

        let mut c_vector = inputs.to_vec();
        c_vector.append(&mut vec![c5, c6]);

        assert_eq!(
            test_circuit.h_polynomial(&c_vector),
            Polynomial::new_monomial(FE::from(18), 0)
        );

        let inputs = [FE::from(4), FE::from(3), FE::from(2), FE::from(1)];

        let (c5, c6) = test_qap_solver(inputs.clone());

        let mut c_vector = inputs.to_vec();
        c_vector.append(&mut vec![c5, c6]);

        assert_eq!(
            test_circuit.h_polynomial(&c_vector),
            Polynomial::new_monomial(FE::from(5), 0)
        );
    }

    #[test]
    fn test_circuit_solver_on_2_2_2_2_outputs_4_and_16() {
        let inputs = [FE::from(2), FE::from(2), FE::from(2), FE::from(2)];

        let (c5, c6) = test_qap_solver(inputs);
        assert_eq!(c5, FE::from(4));
        assert_eq!(c6, FE::from(16));
    }

    #[test]
    fn test_circuit_solver_on_1_2_3_4_outputs_12_and_36() {
        let inputs = [FE::from(1), FE::from(2), FE::from(3), FE::from(4)];

        let (c5, c6) = test_qap_solver(inputs);
        assert_eq!(c5, FE::from(12));
        assert_eq!(c6, FE::from(36));
    }
    #[test]
    fn test_r1cs_into_qap_is_test_qap() {
        let qap = new_test_qap();
        let r1cs = new_test_r1cs();
        let r1cs_as_qap: QuadraticArithmeticProgram = r1cs.into();
        assert_eq!(qap, r1cs_as_qap);
    }
}
