mod commands;
use clap::Parser;
use commands::{MerkleArgs, MerkleEntity};
use lambdaworks_crypto::{
    hash::poseidon::Poseidon,
    merkle_tree::{merkle::MerkleTree, proof::Proof},
};
use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381PrimeField,
    field::element::FieldElement,
    traits::{Deserializable, Serializable},
};
use std::fs;

type FE = FieldElement<BLS12381PrimeField>;

fn generate_merkle_tree(tree_path: String) {
    let values: Vec<FE> = fs::read_to_string(tree_path)
        .expect("Unable to read file")
        .lines()
        .map(FE::from_hex_unchecked)
        .collect();

    let merkle_tree = MerkleTree::<Poseidon<BLS12381PrimeField>>::build(&values);
    let limbs = merkle_tree.root.value().limbs;
    println!("Generated merkle tree with root: {:?}", limbs)

    // save root to file?
}

fn generate_merkle_proof() {
    // create tree from file
    let values: Vec<FE> = fs::read_to_string("sample_tree.csv")
        .unwrap_or_default() // handle error here
        .lines()
        .map(FE::from_hex_unchecked)
        .collect();

    let merkle_tree = MerkleTree::<Poseidon<BLS12381PrimeField>>::build(&values);

    // create proof by pos?
    let proof = merkle_tree.get_proof_by_pos(1).unwrap();

    // serialize proof to file
    let data = proof.serialize();
    fs::write("sample_proof.csv", data).expect("Unable to write file");
}

fn verify_merkle_proof() {
    // read root from file (limbs, format?)
    let root_hash = FE::from_hex_unchecked("0x1");

    // deserialize proof from file
    let bytes = fs::read("sample_proof.csv").expect("Unable to read file");
    let proof: Proof<FE> = Proof::deserialize(&bytes).unwrap();

    proof.verify::<Poseidon<BLS12381PrimeField>>(&root_hash, 1, &FE::from_hex_unchecked("0x1"));
}

fn main() {
    let args: MerkleArgs = MerkleArgs::parse();
    match args.entity {
        MerkleEntity::GenerateMerkleTree(args) => {
            generate_merkle_tree(args.tree_path);
        }
        MerkleEntity::GenerateProof(_args) => generate_merkle_proof(),
        MerkleEntity::VerifyProof(_args) => verify_merkle_proof(),
    }
}
