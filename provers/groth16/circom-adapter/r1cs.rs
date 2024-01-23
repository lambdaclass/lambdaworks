use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrElement as FE;
use lambdaworks_math::unsigned_integer::element::UnsignedInteger;
use serde_json::Value;
use std::fs;

pub type U256 = UnsignedInteger<4>;
fn r1cs() {
    let file_content = fs::read_to_string("src/circuit.r1cs.json")
        .expect("Error reading the file");

    let r1cs_data: Value = serde_json::from_str(&file_content)
        .expect("Error parsing JSON");

    let n_constraints = r1cs_data["nConstraints"].as_u64().unwrap() as usize;
    let n_vars = r1cs_data["nVars"].as_u64().unwrap() as usize;

    let mut matrices = vec![
        vec![vec![FE::zero(); n_vars]; n_constraints], // A matrix
        vec![vec![FE::zero(); n_vars]; n_constraints], // B matrix
        vec![vec![FE::zero(); n_vars]; n_constraints], // C matrix
    ];

    for (matrix_index, matrix) in matrices.iter_mut().enumerate() {
        if let Some(constraint_arrays) = r1cs_data["constraints"].get(matrix_index) {
            for (row_index, sub_array) in constraint_arrays.as_array().unwrap().iter().enumerate() {
                for (key, value) in sub_array.as_object().unwrap() {
                    let col = key.parse::<usize>().unwrap(); // JSON uses 0-based index
                    let u256_value = U256::from_dec_str(value.as_str().unwrap()).unwrap();
                    let hex_str = u256_value.to_hex();
                    let fe_value = FE::from_hex_unchecked(&hex_str);

                    matrix[row_index][col] = fe_value;
                }
            }
        }
    }
    
}