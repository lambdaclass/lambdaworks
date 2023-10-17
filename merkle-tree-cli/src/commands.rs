use clap::{Args, Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(author = "Lambdaworks", version, about)]
pub struct MerkleArgs {
    #[clap(subcommand)]
    pub entity: MerkleEntity,
}

#[derive(Subcommand, Debug)]
pub enum MerkleEntity {
    #[clap(about = "Generate a merkle tree and root")]
    GenerateMerkleTree(GenerateMerkleTreeArgs),
    #[clap(about = "Generate a merkle proof")]
    GenerateProof(GenerateProofArgs),
    #[clap(about = "Verify a merkle proof")]
    VerifyProof(VerifyArgs),
}

#[derive(Args, Debug)]
pub struct GenerateMerkleTreeArgs {
    pub elements: String,
}

#[derive(Args, Debug)]
pub struct GenerateProofArgs {
    pub tree_path: String,
}

#[derive(Args, Debug)]
pub struct VerifyArgs {
    pub proof_path: String,
    pub root_path: String,
}
