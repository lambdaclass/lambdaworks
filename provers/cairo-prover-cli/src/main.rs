use cairo_platinum_prover::air::{generate_cairo_proof, verify_cairo_proof, PublicInputs};
use cairo_platinum_prover::cairo_layout::CairoLayout;
use cairo_platinum_prover::runner::run::generate_prover_args;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
use lambdaworks_math::traits::{Deserializable, Serializable};
use stark_platinum_prover::proof::options::{ProofOptions, SecurityLevel};
use stark_platinum_prover::proof::stark::StarkProof;

use std::time::Instant;

use clap::{Parser, Subcommand, Args};

#[derive(Parser, Debug)]
#[command(author = "Lambdaworks", version, about)]
struct ProverArgs {
    #[clap(subcommand)]
    entity: ProverEntity,
}

#[derive(Subcommand, Debug)]
enum ProverEntity {
    #[clap(about = "Generate a proof for a given compiled cairo program")]
    Prove(ProveArgs),
    #[clap(about = "Verify a proof for a given compiled cairo program")]
    Verify(VerifyArgs),
    #[clap(about = "Generate and verify a proof for a given compiled cairo program")]
    ProveAndVerify(ProveAndVerifyArgs),
}

#[derive(Args, Debug)]
struct ProveArgs {
    input_path: String,
    output_path: String,
}

#[derive(Args, Debug)]
struct VerifyArgs {
    input_path: String,
}

#[derive(Args, Debug)]
struct ProveAndVerifyArgs {
    input_path: String,
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
        ProverEntity::Prove(args) => {
            // verify input file is .cairo
            if args.input_path.contains(".cairo") {
                println!("\nYou are trying to prove a non compiled Cairo program. Please compile it before sending it to the prover.\n");
                return;
            }

            let Some((proof, pub_inputs)) 
                = generate_proof(&args.input_path, &proof_options) 
            else {
                return;
            };

            let mut bytes = vec![];
            let proof_bytes = proof.serialize();
            bytes.extend(proof_bytes.len().to_be_bytes());
            bytes.extend(proof_bytes);
            bytes.extend(pub_inputs.serialize());

            let Ok(()) = std::fs::write(&args.output_path, bytes) else {
                println!("Error writing proof to file: {}", args.output_path);
                return;
            };
            println!("Proof written to {}", args.output_path);
        },
        ProverEntity::Verify(args) => {
            let Ok(program_content) = std::fs::read(&args.input_path) else {
                println!("Error opening {} file", args.input_path);
                return;
            };
            let mut bytes = program_content.as_slice();
            if bytes.len() < 8 {
                println!("Error reading proof from file: {}", args.input_path);
                return;
            }

            let proof_len = usize::from_be_bytes(bytes[0..8].try_into().unwrap());
            bytes = &bytes[8..];
            if bytes.len() < proof_len {
                println!("Error reading proof from file: {}", args.input_path);
                return;
            }
            let Ok(proof) = StarkProof::<Stark252PrimeField>::deserialize(&bytes[0..proof_len])
            else {
                println!("Error reading proof from file: {}", args.input_path);
                return;
            };
            bytes = &bytes[proof_len..];

            let Ok(pub_inputs) = PublicInputs::deserialize(bytes) else {
                println!("Error reading proof from file: {}", args.input_path);
                return;
            };

            verify_proof(proof, pub_inputs, &proof_options);

        },
        ProverEntity::ProveAndVerify(args) => {
            let Some((proof, pub_inputs)) = generate_proof(&args.input_path, &proof_options) else {
                return;
            };
            verify_proof(proof, pub_inputs, &proof_options);
        },
    }
}
