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
use std::{fs, io};

type FE = FieldElement<BLS12381PrimeField>;

fn load_tree_values(tree_path: String) -> Result<Vec<FE>, io::Error> {
    Ok(fs::read_to_string(tree_path)?
        .lines()
        .map(FE::from_hex_unchecked)
        .collect())
}

fn generate_merkle_tree(tree_path: String) -> Result<(), io::Error> {
    let values: Vec<FE> = load_tree_values(tree_path)?;

    let merkle_tree = MerkleTree::<Poseidon<BLS12381PrimeField>>::build(&values);
    let root = merkle_tree.root.representative().to_string();
    println!("Generated merkle tree with root: {:?}", root); // save to file?
    Ok(())
}

fn generate_merkle_proof(tree_path: String, pos: usize) -> Result<(), io::Error> {
    let values: Vec<FE> = load_tree_values(tree_path)?;

    let merkle_tree = MerkleTree::<Poseidon<BLS12381PrimeField>>::build(&values);

    let Some(proof) = merkle_tree.get_proof_by_pos(pos) else {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "Could not generate proof",
        ));
    };

    let data = proof.serialize();
    fs::write("sample_proof.csv", data)
}

fn verify_merkle_proof(root_path: String, pos: usize) -> Result<(), io::Error> {
    let root_hash = FE::from_hex_unchecked(&fs::read_to_string(root_path)?);

    // deserialize proof from file
    let bytes = fs::read("sample_proof.csv").expect("Unable to read file");
    let proof: Proof<FE> = Proof::deserialize(&bytes).map_err(|_| {
        io::Error::new(
            io::ErrorKind::Other,
            "Could not deserialize proof from file",
        )
    })?;

    proof.verify::<Poseidon<BLS12381PrimeField>>(&root_hash, pos, &FE::from_hex_unchecked("0x1"));
    Ok(())
}

fn main() {
    let args: MerkleArgs = MerkleArgs::parse();
    if let Err(e) = match args.entity {
        MerkleEntity::GenerateMerkleTree(args) => generate_merkle_tree(args.tree_path),
        MerkleEntity::GenerateProof(args) => generate_merkle_proof(args.tree_path, args.position),
        MerkleEntity::VerifyProof(args) => verify_merkle_proof(args.root_path, args.position),
    } {
        println!("Error while running command: {:?}", e);
    }
}
