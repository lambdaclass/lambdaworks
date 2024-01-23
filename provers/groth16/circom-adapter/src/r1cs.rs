use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrElement;
use lambdaworks_math::unsigned_integer::element::UnsignedInteger;
use serde_json::Value;
use std::fs;

pub type U256 = UnsignedInteger<4>;

pub fn read_circom_r1cs(path: &str) {
    let file_content = fs::read_to_string(path).expect("Error reading the file");
    let r1cs_data: Value = serde_json::from_str(&file_content).expect("Error parsing JSON");

    let num_of_vars = r1cs_data["nVars"].as_u64().unwrap() as usize;
    let num_of_outputs = r1cs_data["nOutputs"].as_u64().unwrap() as usize;
    let num_of_pub_inputs = r1cs_data["nPubInputs"].as_u64().unwrap() as usize;
    let num_of_private_inputs = r1cs_data["nPrvInputs"].as_u64().unwrap() as usize;

    let num_of_constraints = r1cs_data["nConstraints"].as_u64().unwrap() as usize;

    let mut matrices = vec![
        vec![vec![FrElement::zero(); num_of_vars]; num_of_constraints], // A matrix
        vec![vec![FrElement::zero(); num_of_vars]; num_of_constraints], // B matrix
        vec![vec![FrElement::zero(); num_of_vars]; num_of_constraints], // C matrix
    ];

    for (matrix_index, matrix) in matrices.iter_mut().enumerate() {
        if let Some(constraint_arrays) = r1cs_data["constraints"].get(matrix_index) {
            for (row_index, sub_array) in constraint_arrays.as_array().unwrap().iter().enumerate() {
                for (col, value) in sub_array.as_object().unwrap() {
                    matrix[row_index][col.parse::<usize>().unwrap()] =
                        circom_str_to_lambda_field_element(value.as_str().unwrap());
                }
            }
        }
    }
}

#[inline]
fn circom_str_to_lambda_field_element(value: &str) -> FrElement {
    let u256_value = U256::from_dec_str(value).unwrap();
    let hex_str = u256_value.to_hex();
    FrElement::from_hex_unchecked(&hex_str)
}
