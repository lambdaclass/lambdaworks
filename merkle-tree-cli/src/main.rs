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
};
use std::{
    fs::{self, File},
    io::{self, BufWriter, Write},
};

type FE = FieldElement<BLS12381PrimeField>;

fn load_fe_from_file(file_path: &String) -> Result<FE, io::Error> {
    FE::from_hex(&fs::read_to_string(file_path)?)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{:?}", e)))
}

fn load_tree_values(tree_path: &String) -> Result<Vec<FE>, io::Error> {
    Ok(fs::read_to_string(tree_path)?
        .lines()
        .map(FE::from_hex_unchecked) // remove hex_unchecked for hex
        .collect())
}

fn generate_merkle_tree(tree_path: String) -> Result<(), io::Error> {
    let values: Vec<FE> = load_tree_values(&tree_path)?;

    let merkle_tree = MerkleTree::<Poseidon<BLS12381PrimeField>>::build(&values);
    let root = merkle_tree.root.representative().to_string();
    println!("Generated merkle tree with root: {:?}", root);
    
    let generated_tree_path = tree_path.replace(".csv", ".json");
    let file = File::create(generated_tree_path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, &merkle_tree)?;
    println!("Saved tree to file");
    Ok(())
}

fn generate_merkle_proof(tree_path: String, pos: usize) -> Result<(), io::Error> {
    let values: Vec<FE> = load_tree_values(&tree_path)?;

    let merkle_tree = MerkleTree::<Poseidon<BLS12381PrimeField>>::build(&values);

    let Some(proof) = merkle_tree.get_proof_by_pos(pos) else {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "Could not generate proof",
        ));
    };

    let proof_path = tree_path.replace(".csv", format!("_proof_{pos}.json").as_str());
    let file = File::create(proof_path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, &proof)?;
    writer.flush()
}

fn verify_merkle_proof(
    root_path: String,
    index: usize,
    proof_path: String,
    leaf_path: String,
) -> Result<(), io::Error> {
    let root_hash: FE = load_fe_from_file(&root_path)?;

    let file_str = fs::read_to_string(proof_path)?;
    let proof: Proof<FE> = serde_json::from_str(&file_str)?;

    let leaf: FE = load_fe_from_file(&leaf_path)?;

    match proof.verify::<Poseidon<BLS12381PrimeField>>(&root_hash, index, &leaf) {
        true => println!("Merkle proof verified succesfully"),
        false => println!("Merkle proof failed verifying"),
    }

    Ok(())
}

fn main() {
    let args: MerkleArgs = MerkleArgs::parse();
    if let Err(e) = match args.entity {
        MerkleEntity::GenerateMerkleTree(args) => generate_merkle_tree(args.tree_path),
        MerkleEntity::GenerateProof(args) => generate_merkle_proof(args.tree_path, args.position),
        MerkleEntity::VerifyProof(args) => {
            verify_merkle_proof(args.root_path, args.index, args.proof_path, args.leaf_path)
        }
    } {
        println!("Error while running command: {:?}", e);
    }
}
