use lambdaworks_groth16::{common::*, QuadraticArithmeticProgram};
use lambdaworks_math::polynomial::Polynomial;

#[cfg(test)]
pub fn qap_from_variable_matrices(
    num_of_public_inputs: usize,
    l: &[Vec<FrElement>],
    r: &[Vec<FrElement>],
    o: &[Vec<FrElement>],
) -> QuadraticArithmeticProgram {
    let num_of_total_inputs = l.len();
    assert_eq!(num_of_total_inputs, r.len());
    assert_eq!(num_of_total_inputs, o.len());
    assert!(num_of_total_inputs > 0);
    assert!(num_of_public_inputs <= num_of_total_inputs);

    let num_of_gates = l[0].len();
    let next_power_of_two = num_of_gates.next_power_of_two();
    let pad_zeroes = next_power_of_two - num_of_gates;

    QuadraticArithmeticProgram {
        num_of_public_inputs,
        num_of_gates: next_power_of_two,
        l: build_variable_polynomials(&apply_padding(l, pad_zeroes)),
        r: build_variable_polynomials(&apply_padding(r, pad_zeroes)),
        o: build_variable_polynomials(&apply_padding(o, pad_zeroes)),
    }
}

#[cfg(test)]
pub fn build_variable_polynomials(from_matrix: &[Vec<FrElement>]) -> Vec<Polynomial<FrElement>> {
    from_matrix
        .iter()
        .map(|row| Polynomial::interpolate_fft::<FrField>(row).unwrap())
        .collect()
}

#[cfg(test)]
pub fn apply_padding(columns: &[Vec<FrElement>], pad_zeroes: usize) -> Vec<Vec<FrElement>> {
    columns
        .iter()
        .map(|column| {
            let mut new_column = column.clone();
            new_column.extend(vec![FrElement::zero(); pad_zeroes]);
            new_column
        })
        .collect()
}
