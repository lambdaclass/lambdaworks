#[cfg(test)]
mod integration_tests;

use lambdaworks_groth16::{common::FrElement, QuadraticArithmeticProgram as QAP};
use lambdaworks_math::unsigned_integer::element::UnsignedInteger;
use serde_json::Value;
use std::fs;

pub type U256 = UnsignedInteger<4>;

pub fn circom_r1cs_to_lambda_qap(path: &str) -> QAP {
    let file_content = fs::read_to_string(path).expect("Error reading the file");
    let circom_r1cs: Value = serde_json::from_str(&file_content).expect("Error parsing JSON");

    let num_of_vars = circom_r1cs["nVars"].as_u64().unwrap() as usize; // Includes "1"
    let num_of_pub_inputs = circom_r1cs["nPubInputs"].as_u64().unwrap() as usize + 1; // Compensate for "1"
    let num_of_constraints = circom_r1cs["nConstraints"].as_u64().unwrap() as usize;

    // let num_of_private_inputs = r1cs_data["nPrvInputs"].as_u64().unwrap() as usize;
    // let num_of_outputs = r1cs_data["nOutputs"].as_u64().unwrap() as usize;

    let mut l = vec![vec![FrElement::zero(); num_of_constraints]; num_of_vars];
    let mut r = vec![vec![FrElement::zero(); num_of_constraints]; num_of_vars];
    let mut o = vec![vec![FrElement::zero(); num_of_constraints]; num_of_vars];

    for (constraint_idx, constraint) in circom_r1cs["constraints"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let constraint = constraint.as_array().unwrap();
        for (var_idx, str_val) in constraint[0].as_object().unwrap() {
            l[var_idx.parse::<usize>().unwrap()][constraint_idx] =
                circom_str_to_lambda_field_element(str_val.as_str().unwrap());
        }
        for (var_idx, str_val) in constraint[1].as_object().unwrap() {
            r[var_idx.parse::<usize>().unwrap()][constraint_idx] =
                circom_str_to_lambda_field_element(str_val.as_str().unwrap());
        }
        for (var_idx, str_val) in constraint[2].as_object().unwrap() {
            o[var_idx.parse::<usize>().unwrap()][constraint_idx] =
                circom_str_to_lambda_field_element(str_val.as_str().unwrap());
        }
    }

    QAP::from_variable_matrices(num_of_pub_inputs, &l, &r, &o)
}

pub fn read_circom_witness(path: &str) -> Vec<FrElement> {
    let file_content = fs::read_to_string(path).expect("Error reading the file");
    let circom_witness: Vec<String> =
        serde_json::from_str(&file_content).expect("Error parsing JSON");

    circom_witness
        .iter()
        .map(|num_str| circom_str_to_lambda_field_element(num_str))
        .collect()
}

#[inline]
fn circom_str_to_lambda_field_element(value: &str) -> FrElement {
    let u256_value = U256::from_dec_str(value).unwrap();
    let hex_str = u256_value.to_hex();
    FrElement::from_hex_unchecked(&hex_str)
}
