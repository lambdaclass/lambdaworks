use clap::{Args, Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(author = "Lambdaworks", version, about)]
pub struct MerkleArgs {
    #[command(subcommand)]
    pub entity: MerkleEntity,
}

#[derive(Subcommand, Debug)]
pub enum MerkleEntity {
    #[command(about = "Generate a merkle tree")]
    GenerateTree(GenerateTreeArgs),
    #[command(about = "Generate a merkle proof")]
    GenerateProof(GenerateProofArgs),
    #[command(about = "Verify a merkle proof")]
    VerifyProof(VerifyArgs),
}

#[derive(Args, Debug)]
pub struct GenerateTreeArgs {
    pub tree_path: String,
}

#[derive(Args, Debug)]
pub struct GenerateProofArgs {
    pub tree_path: String,
    pub position: usize,
}

#[derive(Args, Debug)]
pub struct VerifyArgs {
    pub root_path: String,
    pub index: usize,
    pub proof_path: String,
    pub leaf_path: String,
}
