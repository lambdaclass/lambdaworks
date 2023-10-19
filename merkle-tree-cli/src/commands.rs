use clap::{Args, Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(author = "Lambdaworks", version, about)]
pub struct MerkleArgs {
    #[clap(subcommand)]
    pub entity: MerkleEntity,
}

#[derive(Subcommand, Debug)]
pub enum MerkleEntity {
    #[clap(about = "Generate a merkle tree")]
    GenerateMerkleTree(GenerateMerkleTreeArgs),
    #[clap(about = "Generate a merkle proof")]
    GenerateProof(GenerateProofArgs),
    #[clap(about = "Verify a merkle proof")]
    VerifyProof(VerifyArgs),
}

#[derive(Args, Debug)]
pub struct GenerateMerkleTreeArgs {
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
