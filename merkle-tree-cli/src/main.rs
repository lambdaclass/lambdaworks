mod commands;
use clap::Parser;
use commands::{MerkleArgs, MerkleEntity};
use lambdaworks_crypto::{merkle_tree::merkle::MerkleTree, hash::poseidon::Poseidon};
use lambdaworks_math::{elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381PrimeField, field::element::FieldElement};

fn generate_merkle_tree() {
    type FE = FieldElement<BLS12381PrimeField>;

    let values: Vec<FE> = [].to_vec(); // each line of the csv file
    let merkle_tree = MerkleTree::<Poseidon<BLS12381PrimeField>>::build(&values);
}

fn main() {
    let args: MerkleArgs = MerkleArgs::parse();
    match args.entity {
        MerkleEntity::GenerateMerkleTree(_args) => {
            generate_merkle_tree();
        }
        MerkleEntity::GenerateProof(_args) => {
            println!("todo")
        }
        MerkleEntity::VerifyProof(_args) => {
            println!("todo")
        }
    }
}
