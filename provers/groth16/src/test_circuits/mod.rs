use crate::{common::*, QuadraticArithmeticProgram};

/*
Represents x^3 + x + 5 = 35, based on https://vitalik.ca/general/2016/12/10/qap.html
    sym_1 = x * x
    y = sym_1 * x
    sym_2 = y + x
    ~out = sym_2 + 5
*/
pub fn vitalik_qap() -> QuadraticArithmeticProgram {
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

/*
Represents x^2 = 25 or y^2 = 9
    input signal x, y

    sym_1 = x * x -> 25
    sym_2 = y * y -> 9

    sym_3 = sym_1 - 25
    sym_4 = sym_2 - 9

    ~out = sym_3 * sym_4 -> needs to be zero
*///
pub fn test_qap_2() -> QuadraticArithmeticProgram {
    let num_of_public_inputs = 2;
    let [l, r, o] = [
        [
            ["0", "0", "-19", "-9", "0"], //1
            ["1", "0", "0", "0", "0"],    //x
            ["0", "1", "0", "0", "0"],    //y
            ["0", "0", "0", "0", "1"],    //~out
            ["0", "0", "1", "0", "0"],    //sym_1
            ["0", "0", "0", "1", "0"],    //sym_2
            ["0", "0", "0", "0", "1"],    //sym_3
            ["0", "0", "0", "0", "0"],    //sym_4
        ],
        [
            ["0", "0", "1", "1", "0"], //1
            ["1", "0", "0", "0", "0"], //x
            ["0", "1", "0", "0", "0"], //y
            ["0", "0", "0", "0", "0"], //~out
            ["0", "0", "0", "0", "0"], //sym_1
            ["0", "0", "0", "0", "0"], //sym_2
            ["0", "0", "0", "0", "0"], //sym_3
            ["0", "0", "0", "0", "1"], //sym_4
        ],
        [
            ["0", "0", "0", "0", "0"], //1
            ["0", "0", "0", "0", "0"], //x
            ["0", "0", "0", "0", "0"], //y
            ["0", "0", "0", "0", "1"], //~out
            ["1", "0", "0", "0", "0"], //sym_1
            ["0", "1", "0", "0", "0"], //sym_2
            ["0", "0", "1", "0", "0"], //sym_3
            ["0", "0", "0", "1", "0"], //sym_4
        ],
    ]
    .map(|matrix| {
        matrix.map(|row| {
            row.map(|elem| {
                if elem.starts_with("-") {
                    -FrElement::from_hex_unchecked(&elem.chars().skip(1).collect::<String>())
                } else {
                    FrElement::from_hex_unchecked(elem)
                }
            })
            .to_vec()
        })
    });
    QuadraticArithmeticProgram::from_variable_matrices(num_of_public_inputs, &l, &r, &o)
}
