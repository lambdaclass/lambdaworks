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
    #[clap(about = "Generate a proof from a given trace of a cairo program execution")]
    ProveTrace(ProveTraceArgs),
    #[clap(about = "Verify a proof for a given compiled cairo program")]
    Verify(VerifyArgs),
    #[clap(about = "Generate and verify a proof for a given compiled cairo program")]
    ProveAndVerify(ProveAndVerifyArgs),
    #[clap(about = "Compile and prove a given cairo program")]
    CompileAndProve(CompileAndProveArgs),
    #[clap(about = "Compile, prove and verify a given cairo program")]
    CompileProveAndVerify(CompileAndRunAllArgs),
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
pub struct ProveTraceArgs {
    pub trace_bin_path: String,
    pub memory_bin_path: String,
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
