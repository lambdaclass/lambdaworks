use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::FrElement as FE;
use lambdaworks_math::unsigned_integer::element::UnsignedInteger;
use std::fs;


pub type U256 = UnsignedInteger<4>;

fn wtns() {
    let file_content = fs::read_to_string("src/witness.json")
        .expect("Error reading the file");

    let witness_data: Vec<String> = serde_json::from_str(&file_content)
        .expect("Error parsing JSON");

    let witness_fe: Vec<FE> = witness_data.iter()
        .map(|num_str| {
            let u256_value = U256::from_dec_str(num_str).unwrap();
            let hex_str = u256_value.to_hex();
            FE::from_hex_unchecked(&hex_str)
        })
        .collect();

    // Use witness_fe as needed
    for fe in witness_fe {
        println!("{:?}", fe);
    }
}