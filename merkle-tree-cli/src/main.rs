mod commands;
use clap::Parser;
use commands::{MerkleArgs, MerkleEntity};

fn main() {
    let args: MerkleArgs = MerkleArgs::parse();
    match args.entity {
        MerkleEntity::GenerateMerkleTree(_args) => {
            println!("todo")
        }        
        MerkleEntity::GenerateProof(_args) => {
            println!("todo")
        }
        MerkleEntity::VerifyProof(_args) => {
            println!("todo")
        }
    }
}
