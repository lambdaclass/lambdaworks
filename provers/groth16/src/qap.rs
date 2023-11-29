use lambdaworks_math::{fft::polynomial::FFTPoly, polynomial::Polynomial};

use crate::{common::*, r1cs::R1CS};

#[derive(Debug)]
pub struct QuadraticArithmeticProgram {
    pub num_of_public_inputs: usize,
    pub l: Vec<Polynomial<FrElement>>,
    pub r: Vec<Polynomial<FrElement>>,
    pub o: Vec<Polynomial<FrElement>>,
}

impl QuadraticArithmeticProgram {
    pub fn from_r1cs(r1cs: R1CS) -> Self {
        let mut l: Vec<Polynomial<FrElement>> = vec![];
        let mut r: Vec<Polynomial<FrElement>> = vec![];
        let mut o: Vec<Polynomial<FrElement>> = vec![];

        let pad_zeroes =
            r1cs.number_of_constraints().next_power_of_two() - r1cs.number_of_constraints();
        let from_slice = vec![FrElement::zero(); pad_zeroes];

        let mut barsumi_1: Vec<Vec<FrElement>> = vec![];
        let mut barsumi_2: Vec<Vec<FrElement>> = vec![];
        let mut barsumi_3: Vec<Vec<FrElement>> = vec![];

        for i in 0..r1cs.witness_size() {
            let mut l_padded = r1cs
                .constraints
                .iter()
                .map(|c| c.a[i].clone())
                .collect::<Vec<_>>();
            l_padded.extend(from_slice.clone());
            let mut r_padded = r1cs
                .constraints
                .iter()
                .map(|c| c.b[i].clone())
                .collect::<Vec<_>>();
            r_padded.extend(from_slice.clone());
            let mut o_padded = r1cs
                .constraints
                .iter()
                .map(|c| c.c[i].clone())
                .collect::<Vec<_>>();
            o_padded.extend(from_slice.clone());

            barsumi_1.push(l_padded.clone());
            barsumi_2.push(r_padded.clone());
            barsumi_3.push(o_padded.clone());

            l.push(Polynomial::interpolate_fft(&l_padded).unwrap());
            r.push(Polynomial::interpolate_fft(&r_padded).unwrap());
            o.push(Polynomial::interpolate_fft(&o_padded).unwrap());
        }

        println!("--------- L matrix");
        barsumi_1.iter().for_each(|row| {
            row.iter().for_each(|e| print!("{} ", e.to_string()));
            println!();
        });

        println!("--------- R matrix");
        barsumi_2.iter().for_each(|row| {
            row.iter().for_each(|e| print!("{} ", e.to_string()));
            println!();
        });

        println!("--------- O matrix");
        barsumi_3.iter().for_each(|row| {
            row.iter().for_each(|e| print!("{} ", e.to_string()));
            println!();
        });

        QuadraticArithmeticProgram {
            l,
            r,
            o,
            num_of_public_inputs: r1cs.number_of_inputs,
        }
    }

    pub fn from_variable_matrices(
        num_of_public_inputs: usize,
        l: &[Vec<FrElement>],
        r: &[Vec<FrElement>],
        o: &[Vec<FrElement>],
    ) -> Self {
        let num_of_total_inputs = l.len();
        assert_eq!(num_of_total_inputs, r.len());
        assert_eq!(num_of_total_inputs, o.len());
        assert!(num_of_total_inputs > 0);
        assert!(num_of_public_inputs <= num_of_total_inputs);

        let num_of_gates = l[0].len();
        let pad_zeroes = num_of_gates.next_power_of_two() - num_of_gates;
        let l = Self::apply_padding(l, pad_zeroes);
        let r = Self::apply_padding(r, pad_zeroes);
        let o = Self::apply_padding(o, pad_zeroes);

        Self {
            num_of_public_inputs,
            l: Self::build_variable_polynomials(&l),
            r: Self::build_variable_polynomials(&r),
            o: Self::build_variable_polynomials(&o),
        }
    }

    pub fn num_of_gates(&self) -> usize {
        self.l[0].degree() + 1
    }

    pub fn num_of_private_inputs(&self) -> usize {
        self.l.len() - self.num_of_public_inputs
    }

    pub fn num_of_total_inputs(&self) -> usize {
        self.l.len()
    }

    pub fn calculate_h_coefficients(&self, w: &[FrElement]) -> Vec<FrElement> {
        let offset = &ORDER_R_MINUS_1_ROOT_UNITY;
        let degree = self.num_of_gates() * 2;

        let [l, r, o] = self.scale_and_accumulate_variable_polynomials(w, degree, offset);

        // TODO: Change to a vector of offsetted evaluations of x^N-1
        let mut t = (Polynomial::new_monomial(FrElement::one(), self.num_of_gates())
            - FrElement::one())
        .evaluate_offset_fft(1, Some(degree), offset)
        .unwrap();
        FrElement::inplace_batch_inverse(&mut t).unwrap();

        let h_evaluated = l
            .iter()
            .zip(&r)
            .zip(&o)
            .zip(&t)
            .map(|(((l, r), o), t)| (l * r - o) * t)
            .collect::<Vec<_>>();

        Polynomial::interpolate_offset_fft(&h_evaluated, offset)
            .unwrap()
            .coefficients()
            .to_vec()
    }

    fn apply_padding(columns: &[Vec<FrElement>], pad_zeroes: usize) -> Vec<Vec<FrElement>> {
        let from_slice = vec![FrElement::zero(); pad_zeroes];
        columns
            .iter()
            .map(|column| {
                let mut new_column = column.clone();
                new_column.extend_from_slice(&from_slice);
                new_column
            })
            .collect::<Vec<_>>()
    }

    fn build_variable_polynomials(from_matrix: &[Vec<FrElement>]) -> Vec<Polynomial<FrElement>> {
        from_matrix
            .iter()
            .map(|row| Polynomial::interpolate_fft(row).unwrap())
            .collect()
    }

    // Compute A.s by summing up polynomials A[0].s, A[1].s, ..., A[n].s
    // In other words, assign the witness coefficients / execution values
    // Similarly for B.s and C.s
    fn scale_and_accumulate_variable_polynomials(
        &self,
        w: &[FrElement],
        degree: usize,
        offset: &FrElement,
    ) -> [Vec<FrElement>; 3] {
        [&self.l, &self.r, &self.o].map(|var_polynomials| {
            var_polynomials
                .iter()
                .zip(w)
                .map(|(poly, coeff)| poly.mul_with_ref(&Polynomial::new_monomial(coeff.clone(), 0)))
                .reduce(|poly1, poly2| poly1 + poly2)
                .unwrap()
                .evaluate_offset_fft(1, Some(degree), offset)
                .unwrap()
        })
    }
}
