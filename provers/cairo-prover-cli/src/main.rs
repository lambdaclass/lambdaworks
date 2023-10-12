use cairo_platinum_prover::air::{generate_cairo_proof, verify_cairo_proof, PublicInputs};
use cairo_platinum_prover::cairo_layout::CairoLayout;
use cairo_platinum_prover::runner::run::generate_prover_args;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
use lambdaworks_math::traits::{Deserializable, Serializable};
use stark_platinum_prover::proof::options::{ProofOptions, SecurityLevel};
use stark_platinum_prover::proof::stark::StarkProof;

use std::env;
use std::fs::File;
use std::io::{Error, ErrorKind};
use std::process::Command;
use std::time::Instant;

use clap::{Args, Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(author = "Lambdaworks", version, about)]
struct ProverArgs {
    #[clap(subcommand)]
    entity: ProverEntity,
}

#[derive(Subcommand, Debug)]
enum ProverEntity {
    #[clap(about = "Compile a given cairo program")]
    Compile(CompileArgs),
    #[clap(about = "Generate a proof for a given compiled cairo program")]
    Prove(ProveArgs),
    #[clap(about = "Verify a proof for a given compiled cairo program")]
    Verify(VerifyArgs),
    #[clap(about = "Generate and verify a proof for a given compiled cairo program")]
    ProveAndVerify(ProveAndVerifyArgs),
}

#[derive(Args, Debug)]
struct CompileArgs {
    program_path: String,
}

#[derive(Args, Debug)]
struct ProveArgs {
    program_path: String,
    proof_path: String,
}

#[derive(Args, Debug)]
struct VerifyArgs {
    proof_path: String,
}

#[derive(Args, Debug)]
struct ProveAndVerifyArgs {
    program_path: String,
}

/// Get current directory and return it as a String
fn get_root_dir() -> Result<String, Error> {
    let path_buf = env::current_dir()?.canonicalize()?;
    if let Some(path) = path_buf.to_str() {
        return Ok(path.to_string());
    }

    Err(Error::new(ErrorKind::NotFound, "not found"))
}

/// Attemps to compile the Cairo program with `cairo-compile`
/// and then save it to the desired path.  
/// Returns `Ok` on success else returns `Error`
fn cairo_compile(program_path: &String, out_file_path: &String) -> Result<(), Error> {
    let out_file = File::create(out_file_path)?;

    if let Err(err) = Command::new("cairo-compile")
        .arg("--proof_mode")
        .arg(program_path)
        .stdout(out_file)
        .spawn()
    {
        return Err(err);
    }

    Ok(())
}

/// Attemps to compile the Cairo program with `docker`
/// and then save it to the desired path.  
/// Returns `Ok` on success else returns `Error`
fn docker_compile(program_path: &String, out_file_path: &String) -> Result<(), Error> {
    let out_file = File::create(out_file_path)?;
    let root_dir = get_root_dir()?;
    if let Err(err) = Command::new("docker")
        .arg("run")
        .arg("-v")
        .arg(format!("{}/:/pwd", root_dir))
        .arg("cairo")
        .arg("--proof_mode")
        .arg(format!("/pwd/{}", program_path))
        .stdout(out_file)
        .spawn()
    {
        return Err(err);
    }

    Ok(())
}

fn generate_proof(
    input_path: &String,
    proof_options: &ProofOptions,
) -> Option<(StarkProof<Stark252PrimeField>, PublicInputs)> {
    let timer = Instant::now();

    let Ok(program_content) = std::fs::read(input_path) else {
        println!("Error opening {input_path} file");
        return None;
    };

    // FIXME: We should set this through the CLI in the future
    let layout = CairoLayout::Plain;

    let Ok((main_trace, pub_inputs)) = generate_prover_args(&program_content, &None, layout) else {
        println!("Error generating prover args");
        return None;
    };

    println!("  Time spent: {:?} \n", timer.elapsed());

    let timer = Instant::now();
    println!("Making proof ...");
    let proof = match generate_cairo_proof(&main_trace, &pub_inputs, proof_options) {
        Ok(p) => p,
        Err(e) => {
            println!("Error generating proof: {:?}", e);
            return None;
        }
    };

    println!("Time spent in proving: {:?} \n", timer.elapsed());

    Some((proof, pub_inputs))
}

fn verify_proof(
    proof: StarkProof<Stark252PrimeField>,
    pub_inputs: PublicInputs,
    proof_options: &ProofOptions,
) -> bool {
    let timer = Instant::now();

    println!("Verifying ...");
    let proof_verified = verify_cairo_proof(&proof, &pub_inputs, proof_options);
    println!("Time spent in verifying: {:?} \n", timer.elapsed());

    if proof_verified {
        println!("Verification succeded");
    } else {
        println!("Verification failed");
    }

    proof_verified
}

fn main() {
    let proof_options = ProofOptions::new_secure(SecurityLevel::Conjecturable100Bits, 3);

    let args: ProverArgs = ProverArgs::parse();
    match args.entity {
        ProverEntity::Compile(args) => {
            let out_file_path = args.program_path.replace(".cairo", ".json");

            if cairo_compile(&args.program_path, &out_file_path).is_ok()
                || docker_compile(&args.program_path, &out_file_path).is_ok()
            {
                println!("Compiled cairo program");
            } else {
                println!("Failed to compile cairo program\nEnsure cairo-compile or docker is installed (and running)")
            }
        }
        ProverEntity::Prove(args) => {
            // verify input file is .cairo
            if args.program_path.contains(".cairo") {
                println!("\nYou are trying to prove a non compiled Cairo program. Please compile it before sending it to the prover.\n");
                return;
            }

            let Some((proof, pub_inputs)) = generate_proof(&args.program_path, &proof_options)
            else {
                return;
            };

            let mut bytes = vec![];
            let proof_bytes = proof.serialize();
            bytes.extend(proof_bytes.len().to_be_bytes());
            bytes.extend(proof_bytes);
            bytes.extend(pub_inputs.serialize());

            let Ok(()) = std::fs::write(&args.proof_path, bytes) else {
                println!("Error writing proof to file: {}", args.proof_path);
                return;
            };
            println!("Proof written to {}", args.proof_path);
        }
        ProverEntity::Verify(args) => {
            let Ok(program_content) = std::fs::read(&args.proof_path) else {
                println!("Error opening {} file", args.proof_path);
                return;
            };
            let mut bytes = program_content.as_slice();
            if bytes.len() < 8 {
                println!("Error reading proof from file: {}", args.proof_path);
                return;
            }

            let proof_len = usize::from_be_bytes(bytes[0..8].try_into().unwrap());
            bytes = &bytes[8..];
            if bytes.len() < proof_len {
                println!("Error reading proof from file: {}", args.proof_path);
                return;
            }
            let Ok(proof) = StarkProof::<Stark252PrimeField>::deserialize(&bytes[0..proof_len])
            else {
                println!("Error reading proof from file: {}", args.proof_path);
                return;
            };
            bytes = &bytes[proof_len..];

            let Ok(pub_inputs) = PublicInputs::deserialize(bytes) else {
                println!("Error reading proof from file: {}", args.proof_path);
                return;
            };

            verify_proof(proof, pub_inputs, &proof_options);
        }
        ProverEntity::ProveAndVerify(args) => {
            let Some((proof, pub_inputs)) = generate_proof(&args.program_path, &proof_options)
            else {
                return;
            };
            verify_proof(proof, pub_inputs, &proof_options);
        }
    }
}
