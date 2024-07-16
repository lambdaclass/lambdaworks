//use super::super::config::ORDER_R;
//use super::r1cs::R1CS;
use lambdaworks_math::polynomial::Polynomial;
use crate::{common::*, r1cs::R1CS};
//use crate::math::{field_element::FieldElement, polynomial::Polynomial as Poly};
use std::convert::From;

// type FE = FieldElement<ORDER_R>;
// type Polynomial = Polynomial<FE>;

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
        // TO DO: Check if the amount of inputs and outputs matches the polynomials
        if vs.len() != ws.len() || vs.len() != ys.len() || ws.len() != ys.len() {
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
    /// Receives C elements of a solution of the circuit
    /// Returns p polynomial
    // This along the polynomial execution should be migrated with a better
    // representation of the circuit
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
    // Check R1CS
    /// Transforms a R1CS to a QAP
    fn from(r1cs: R1CS) -> Self {
        // The r values for the qap polynomial can each be any number,
        // as long as there are the right amount of rs
        // In this case, it's set them to be 0,1,2..number_of_constraints(),
        // number_of_constraints non inclusive
        let rs: Vec<FE> = (0..r1cs.number_of_constraints() as u128)
            .map(|i| FE::new(i.into()))
            .collect();
// why .into()?

// in the pinocchio_lambda_vs implementation we found:
/*
        let rs: Vec<FE> = (0..r1cs.number_of_constraints() as u128)
            .map(FE::new)
            .collect();

*/
        let mut vs: Vec<Polynomial<FE>> = Vec::with_capacity(r1cs.witness_size());
        let mut ws: Vec<Polynomial<FE>> = Vec::with_capacity(r1cs.witness_size());
        let mut ys: Vec<Polynomial<FE>> = Vec::with_capacity(r1cs.witness_size());
        let mut t: Polynomial<FE> = Polynomial::new_monomial(FE::from(1), 0);
//  (x-rs)
// To do: Check Polynomial::new()
        for r in &rs {
            t = t * Polynomial::new(&vec![-r, FE::from(1)]);
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


