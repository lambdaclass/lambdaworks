use clap::{Args, Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(author = "Lambdaworks", version, about)]
pub struct ProverArgs {
    #[clap(subcommand)]
    pub entity: ProverEntity,
}

#[derive(Subcommand, Debug)]
pub enum ProverEntity {
    #[clap(about = "Compile a given cairo program")]
    Compile(CompileArgs),
    #[clap(about = "Generate a proof for a given compiled cairo program")]
    Prove(ProveArgs),
    #[clap(about = "Verify a proof for a given compiled cairo program")]
    Verify(VerifyArgs),
    #[clap(about = "Generate and verify a proof for a given compiled cairo program")]
    ProveAndVerify(ProveAndVerifyArgs),
    #[clap(about = "Compile and prove a given cairo program")]
    CompileAndProve(CompileAndProveArgs),
    #[clap(about = "Compile, prove and verify a given cairo program")]
    CompileAndRunAll(CompileAndRunAllArgs)
}

#[derive(Args, Debug)]
pub struct CompileArgs {
    pub program_path: String,
}

#[derive(Args, Debug)]
pub struct ProveArgs {
    pub program_path: String,
    pub proof_path: String,
}

#[derive(Args, Debug)]
pub struct VerifyArgs {
    pub proof_path: String,
}

#[derive(Args, Debug)]
pub struct ProveAndVerifyArgs {
    pub program_path: String,
}

#[derive(Args, Debug)]
pub struct CompileAndProveArgs {
    pub program_path: String,
    pub proof_path: String,
}

#[derive(Args, Debug)]
pub struct CompileAndRunAllArgs {
    pub program_path: String,
}
