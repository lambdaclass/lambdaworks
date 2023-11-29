use std::collections::{HashMap, HashSet};

use lambdaworks_math::{fft::polynomial::FFTPoly, polynomial::Polynomial};

use crate::common::*;

#[derive(Debug)]
pub struct QuadraticArithmeticProgram {
    pub num_of_public_inputs: usize,
    pub l: Vec<Polynomial<FrElement>>,
    pub r: Vec<Polynomial<FrElement>>,
    pub o: Vec<Polynomial<FrElement>>,
}

impl QuadraticArithmeticProgram {
    pub fn from_constraints(csv: &str, public_inputs: &HashSet<&str>) -> Self {
        let constraints = csv
            .split([',', '\n'])
            .filter(|elem| elem.len() > 0)
            .map(|term| {
                term.split(['+', ' '])
                    .filter(|elem| elem.len() > 0)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let constraints = constraints.chunks(3).collect::<Vec<_>>();

        let mut l_vars: HashMap<&str, Vec<&str>> = HashMap::new(); // var -> QAP row
        let mut r_vars: HashMap<&str, Vec<&str>> = HashMap::new();
        let mut o_vars: HashMap<&str, Vec<&str>> = HashMap::new();

        for (gate_idx, row) in constraints.iter().enumerate() {
            println!("{:?}", row);
            for var in row[0].iter() {
                let is_constant = var.parse::<u8>().is_ok();
                let key = if is_constant { "1" } else { var };
                l_vars
                    .entry(key)
                    .or_insert_with(|| vec!["0"; constraints.len()])[gate_idx] =
                    if is_constant { var } else { "1" };
            }
            for var in row[1].iter() {
                let is_constant = var.parse::<u8>().is_ok();
                let key = if is_constant { "1" } else { var };
                r_vars
                    .entry(key)
                    .or_insert_with(|| vec!["0"; constraints.len()])[gate_idx] =
                    if is_constant { var } else { "1" };
            }
            for var in row[2].iter() {
                let is_constant = var.parse::<u8>().is_ok();
                let key = if is_constant { "1" } else { var };
                o_vars
                    .entry(key)
                    .or_insert_with(|| vec!["0"; constraints.len()])[gate_idx] =
                    if is_constant { var } else { "1" };
            }
        }

        let mut all_vars = HashSet::<&str>::new();
        all_vars.extend(l_vars.keys());
        all_vars.extend(r_vars.keys());
        all_vars.extend(o_vars.keys());

        let mut l_matrix: Vec<Vec<&str>> = Vec::with_capacity(all_vars.len());
        let mut r_matrix: Vec<Vec<&str>> = Vec::with_capacity(all_vars.len());
        let mut o_matrix: Vec<Vec<&str>> = Vec::with_capacity(all_vars.len());

        println!("Constructing matrices - public inputs");
        for var in public_inputs {
            println!("------ {} ------", var);

            l_matrix.push(match l_vars.get(var) {
                Some(vec) => vec.clone(),
                None => vec!["0"; constraints.len()],
            });
            r_matrix.push(match r_vars.get(var) {
                Some(vec) => vec.clone(),
                None => vec!["0"; constraints.len()],
            });
            o_matrix.push(match o_vars.get(var) {
                Some(vec) => vec.clone(),
                None => vec!["0"; constraints.len()],
            });
        }
        println!("Constructing matrices - private inputs");
        for var in all_vars.iter().filter(|var| !public_inputs.contains(*var)) {
            println!("------ {} ------", var);

            l_matrix.push(match l_vars.get(var) {
                Some(vec) => vec.clone(),
                None => vec!["0"; constraints.len()],
            });
            r_matrix.push(match r_vars.get(var) {
                Some(vec) => vec.clone(),
                None => vec!["0"; constraints.len()],
            });
            o_matrix.push(match o_vars.get(var) {
                Some(vec) => vec.clone(),
                None => vec!["0"; constraints.len()],
            });
        }

        let l_matrix = l_matrix
            .iter()
            .map(|row| {
                row.iter()
                    .map(|cell| FrElement::from_hex_unchecked(cell))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let r_matrix = r_matrix
            .iter()
            .map(|row| {
                row.iter()
                    .map(|cell| FrElement::from_hex_unchecked(cell))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let o_matrix = o_matrix
            .iter()
            .map(|row| {
                row.iter()
                    .map(|cell| FrElement::from_hex_unchecked(cell))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        Self::from_variable_matrices(public_inputs.len(), &l_matrix, &r_matrix, &o_matrix)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn import_1() {
        let csv = include_str!("test_circuits/constraint_csv/1.csv");
        let mut inputs = HashSet::new();
        inputs.extend(["1", "x", "~out"]);
        QuadraticArithmeticProgram::from_constraints(csv, &inputs);
    }
}
