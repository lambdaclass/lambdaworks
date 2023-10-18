mod commands;
use clap::Parser;
use commands::{MerkleArgs, MerkleEntity};
use lambdaworks_crypto::{hash::poseidon::Poseidon, merkle_tree::merkle::MerkleTree};
use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381PrimeField,
    field::element::FieldElement,
};
use std::fs;

fn generate_merkle_tree(tree_path: String) {
    type FE = FieldElement<BLS12381PrimeField>;

    let values: Vec<FE> = fs::read_to_string(tree_path)
        .unwrap_or_default() // handle error here
        .lines()
        .map(FE::from_hex_unchecked)
        .collect();

    let merkle_tree = MerkleTree::<Poseidon<BLS12381PrimeField>>::build(&values);
    let limbs = merkle_tree.root.value().limbs;
    println!("Generated merkle tree with root: {:?}", limbs)
}

fn generate_merkle_proof() {
    // create tree from file
    // create proof by pos?
    // serialize proof to file
}

fn verify_merkle_proof() {
    // deserialize proof from file
    // read root from file (limbs, format?)
    // proof.verify::<Poseidon<BLS12381PrimeField>>(&merkle_tree.root, 1, &FE::new(2)));
    // 
}

fn main() {
    let args: MerkleArgs = MerkleArgs::parse();
    match args.entity {
        MerkleEntity::GenerateMerkleTree(args) => {
            generate_merkle_tree(args.tree_path);
        }
        MerkleEntity::GenerateProof(_args) => {
            generate_merkle_proof()
        }
        MerkleEntity::VerifyProof(_args) => {
            verify_merkle_proof()
        }
    }
}
