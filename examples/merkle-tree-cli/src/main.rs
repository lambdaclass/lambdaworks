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
#[cfg(feature = "verbose")]
use std::io::BufWriter;
#[cfg(not(feature = "verbose"))]
use std::io::Read;
use std::{
    fs::{self, File},
    io::{self, Write},
};

type FE = FieldElement<BLS12381PrimeField>;

fn load_fe_from_file(file_path: &String) -> Result<FE, io::Error> {
    FE::from_hex(&fs::read_to_string(file_path)?.replace('\n', ""))
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{:?}", e)))
}

fn load_tree_values(tree_path: &String) -> Result<Vec<FE>, io::Error> {
    Ok(fs::read_to_string(tree_path)?
        .split(';')
        .map(FE::from_hex_unchecked)
        .collect())
}

fn generate_merkle_tree(tree_path: String) -> Result<(), io::Error> {
    let values: Vec<FE> = load_tree_values(&tree_path)?;

    let merkle_tree = MerkleTree::<Poseidon<BLS12381PrimeField>>::build(&values);
    let root = merkle_tree.root.representative().to_string();
    println!("Generated merkle tree with root: {:?}", root);

    #[cfg(feature = "verbose")]
    {
        let generated_tree_path = tree_path.replace(".csv", ".json");
        let file = File::create(generated_tree_path)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, &merkle_tree)?;
    }
    #[cfg(feature = "binary")]
    {
        let generated_tree_path = tree_path.replace(".csv", ".tree");
        let mut file = File::create(generated_tree_path)?;
        let bytes =
            bincode::serde::encode_to_vec(&merkle_tree, bincode::config::standard()).unwrap(); // handle unwrap
        file.write_all(&bytes)?;
    }
    println!("Saved tree to file");
    Ok(())
}

fn generate_merkle_proof(tree_path: String, pos: usize) -> Result<(), io::Error> {
    let values: Vec<FE> = load_tree_values(&tree_path)?;
    let merkle_tree = MerkleTree::<Poseidon<BLS12381PrimeField>>::build(&values);

    let Some(proof) = merkle_tree.get_proof_by_pos(pos) else {
        return Err(io::Error::new(io::ErrorKind::Other, "Index out of bounds"));
    };

    #[cfg(feature = "verbose")]
    {
        let proof_path = tree_path.replace(".csv", format!("_proof_{pos}.json").as_str());
        let file = File::create(proof_path)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, &proof)?;
        writer.flush()
    }
    #[cfg(feature = "binary")]
    {
        let proof_path = tree_path.replace(".csv", format!("_proof_{pos}.proof").as_str());
        let mut file = File::create(proof_path)?;
        let bytes = bincode::serde::encode_to_vec(proof, bincode::config::standard()).unwrap(); // handle unwrap
        file.write_all(&bytes)
    }
}

fn verify_merkle_proof(
    root_path: String,
    index: usize,
    proof_path: String,
    leaf_path: String,
) -> Result<(), io::Error> {
    let root_hash: FE = load_fe_from_file(&root_path)?;

    let leaf: FE = load_fe_from_file(&leaf_path)?;

    #[cfg(feature = "verbose")]
    {
        let file_str = fs::read_to_string(proof_path)?;
        let proof: Proof<FE> = serde_json::from_str(&file_str)?;

        match proof.verify::<Poseidon<BLS12381PrimeField>>(&root_hash, index, &leaf) {
            true => println!("\x1b[32mMerkle proof verified succesfully\x1b[0m"),
            false => println!("\x1b[31mMerkle proof failed verifying\x1b[0m"),
        }
    }
    #[cfg(feature = "binary")]
    {
        let mut file = File::open(proof_path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let (proof, _): (Proof<FE>, _) =
            bincode::serde::decode_from_slice(&buffer, bincode::config::standard()).unwrap();
        // handle unwrap

        match proof.verify::<Poseidon<BLS12381PrimeField>>(&root_hash, index, &leaf) {
            true => println!("\x1b[32mMerkle proof verified succesfully\x1b[0m"),
            false => println!("\x1b[31mMerkle proof failed verifying\x1b[0m"),
        }
    }

    Ok(())
}

fn main() {
    let args: MerkleArgs = MerkleArgs::parse();
    if let Err(e) = match args.entity {
        MerkleEntity::GenerateTree(args) => generate_merkle_tree(args.tree_path),
        MerkleEntity::GenerateProof(args) => generate_merkle_proof(args.tree_path, args.position),
        MerkleEntity::VerifyProof(args) => {
            verify_merkle_proof(args.root_path, args.index, args.proof_path, args.leaf_path)
        }
    } {
        println!("Error while running command: {:?}", e);
    }
}
