use crate::{common::*, QuadraticArithmeticProgram};

/*
Represents x^3 + x + 5 = 35, based on https://vitalik.ca/general/2016/12/10/qap.html
    sym_1 = x * x
    y = sym_1 * x
    sym_2 = y + x
    ~out = sym_2 + 5
*/
pub fn qap_example_circuit_1() -> QuadraticArithmeticProgram {
    let num_of_public_inputs = 1;
    let [l, r, o] = [
        [
            ["0", "0", "0", "5"],
            ["1", "0", "1", "0"],
            ["0", "0", "0", "0"],
            ["0", "1", "0", "0"],
            ["0", "0", "1", "0"],
            ["0", "0", "0", "1"],
        ],
        [
            ["0", "0", "1", "1"],
            ["1", "1", "0", "0"],
            ["0", "0", "0", "0"],
            ["0", "0", "0", "0"],
            ["0", "0", "0", "0"],
            ["0", "0", "0", "0"],
        ],
        [
            ["0", "0", "0", "0"],
            ["0", "0", "0", "0"],
            ["0", "0", "0", "1"],
            ["1", "0", "0", "0"],
            ["0", "1", "0", "0"],
            ["0", "0", "1", "0"],
        ],
    ]
    .map(|matrix| matrix.map(|row| row.map(FrElement::from_hex_unchecked).to_vec()));
    QuadraticArithmeticProgram::from_variable_matrices(num_of_public_inputs, &l, &r, &o)
}
