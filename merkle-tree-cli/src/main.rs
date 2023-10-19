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
use std::{fs::{self, File}, io::{self, BufWriter, Write, BufReader}};

type FE = FieldElement<BLS12381PrimeField>;

fn load_tree_values(tree_path: &String) -> Result<Vec<FE>, io::Error> {
    Ok(fs::read_to_string(tree_path)?
        .lines()
        .map(FE::from_hex_unchecked)
        .collect())
}

fn generate_merkle_tree(tree_path: String) -> Result<(), io::Error> {
    let values: Vec<FE> = load_tree_values(&tree_path)?;

    let merkle_tree = MerkleTree::<Poseidon<BLS12381PrimeField>>::build(&values);
    let root = merkle_tree.root.representative().to_string();
    println!("Generated merkle tree with root: {:?}", root); // save to file?
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
    _root_path: String,
    _index: usize,
    proof_path: String,
) -> Result<(), io::Error> {
    // let _root_hash = FE::from_hex_unchecked(&fs::read_to_string(root_path)?);

    // Open the file in read-only mode with buffer.
    let file = File::open(proof_path)?;
    let reader = BufReader::new(file);

    // Read the JSON contents of the file as an instance of `User`.
    let _proof: Proof<FE> = serde_json::from_reader(reader)?;

    // value is the leaf we start from
    // let res = proof.verify::<Poseidon<BLS12381PrimeField>>(
    //     &root_hash,
    //     index,
    //     &FE::from_hex_unchecked("0x12345"),
    // );
    // println!("Proof verified: {:?}", res);
    Ok(())
}

fn main() {
    let args: MerkleArgs = MerkleArgs::parse();
    if let Err(e) = match args.entity {
        MerkleEntity::GenerateMerkleTree(args) => generate_merkle_tree(args.tree_path),
        MerkleEntity::GenerateProof(args) => generate_merkle_proof(args.tree_path, args.position),
        MerkleEntity::VerifyProof(args) => {
            verify_merkle_proof(args.root_path, args.index, args.proof_path)
        }
    } {
        println!("Error while running command: {:?}", e);
    }
}
