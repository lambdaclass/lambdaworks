use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
use lambdaworks_math::traits::{Deserializable, Serializable};
use platinum_prover::air::{generate_cairo_proof, verify_cairo_proof, PublicInputs};
use platinum_prover::cairo_layout::CairoLayout;
use platinum_prover::runner::run::generate_prover_args;
use stark_platinum_prover::proof::options::{ProofOptions, SecurityLevel};
use stark_platinum_prover::proof::stark::StarkProof;
mod commands;
use clap::Parser;

use std::env;
use std::fs::File;
use std::io::{Error, ErrorKind};
use std::process::{Command, Stdio};
use std::time::Instant;

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

    match Command::new("cairo-compile")
        .arg("--proof_mode")
        .arg(program_path)
        .stderr(Stdio::null())
        .stdout(out_file)
        .spawn()
    {
        Ok(mut child) => {
            // wait for spawned proccess to finish
            match child.wait() {
                Ok(_) => Ok(()),
                Err(err) => Err(err),
            }
        }
        Err(err) => Err(err),
    }
}

/// Attemps to compile the Cairo program with `docker`
/// and then save it to the desired path.  
/// Returns `Ok` on success else returns `Error`
fn docker_compile(program_path: &String, out_file_path: &String) -> Result<(), Error> {
    let out_file = File::create(out_file_path)?;
    let root_dir = get_root_dir()?;
    match Command::new("docker")
        .arg("run")
        .arg("--rm")
        .arg("-v")
        .arg(format!("{}/:/pwd", root_dir))
        .arg("cairo")
        .arg("--proof_mode")
        .arg(format!("/pwd/{}", program_path))
        .stderr(Stdio::null())
        .stdout(out_file)
        .spawn()
    {
        Ok(mut child) => {
            // wait for spawned proccess to finish
            match child.wait() {
                Ok(status) => match status.code() {
                    Some(0) => Ok(()), // exit success
                    _ => Err(Error::new(
                        ErrorKind::Other,
                        "File provided is not a Cairo uncompiled",
                    )),
                },
                Err(err) => Err(err),
            }
        }
        Err(err) => Err(err),
    }
}

/// Attemps to compile the Cairo program
/// with either `cairo-compile` or `docker``
fn try_compile(program_path: &String, out_file_path: &String) -> Result<(), Error> {
    if !program_path.contains(".cairo") {
        return Err(Error::new(
            ErrorKind::Other,
            "Provided file is not a Cairo source file",
        ));
    }

    if cairo_compile(program_path, out_file_path).is_ok()
        || docker_compile(program_path, out_file_path).is_ok()
    {
        Ok(())
    } else {
        Err(Error::new(
            ErrorKind::Other,
            "Failed to compile cairo program, neither cairo-compile nor docker found",
        ))
    }
}

fn generate_proof(
    input_path: &String,
    proof_options: &ProofOptions,
) -> Option<(StarkProof<Stark252PrimeField>, PublicInputs)> {
    let timer = Instant::now();

    let Ok(program_content) = std::fs::read(input_path) else {
        eprintln!("Error opening {input_path} file");
        return None;
    };

    // FIXME: We should set this through the CLI in the future
    let layout = CairoLayout::Plain;

    let Ok((main_trace, pub_inputs)) = generate_prover_args(&program_content, &None, layout) else {
        eprintln!("Error generating prover args");
        return None;
    };

    println!("  Time spent: {:?} \n", timer.elapsed());

    let timer = Instant::now();
    println!("Making proof ...");
    let proof = match generate_cairo_proof(&main_trace, &pub_inputs, proof_options) {
        Ok(p) => p,
        Err(err) => {
            eprintln!("Error generating proof: {:?}", err);
            return None;
        }
    };

    println!("  Time spent in proving: {:?} \n", timer.elapsed());

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
    println!("  Time spent in verifying: {:?} \n", timer.elapsed());

    if proof_verified {
        println!("Verification succeded");
    } else {
        println!("Verification failed");
    }

    proof_verified
}

fn write_proof(
    proof: StarkProof<Stark252PrimeField>,
    pub_inputs: PublicInputs,
    proof_path: String,
) {
    let mut bytes = vec![];

    // cbor serialization (original)
    let proof_bytes: Vec<u8> = serde_cbor::ser::to_vec(&proof).unwrap();
    println!("Cbor: proof size: {} bytes", proof_bytes.len());

    // rmp serialization
    let proof_bytes: Vec<u8> = rmp_serde::to_vec(&proof).unwrap();
    println!("Rmp: proof size: {} bytes", proof_bytes.len());

    // bincode serialization
    let proof_bytes: Vec<u8> =
        bincode::serde::encode_to_vec(&proof, bincode::config::standard()).unwrap();
    println!("Bincode: proof size: {} bytes", proof_bytes.len());

    bytes.extend(proof_bytes.len().to_be_bytes());
    bytes.extend(proof_bytes);
    bytes.extend(pub_inputs.serialize());

    let Ok(()) = std::fs::write(&proof_path, bytes) else {
        eprintln!("Error writing proof to file: {}", &proof_path);
        return;
    };
    println!("Proof written to {}", &proof_path);
}

fn main() {
    let proof_options = ProofOptions::new_secure(SecurityLevel::Conjecturable100Bits, 3);

    let args: commands::ProverArgs = commands::ProverArgs::parse();
    match args.entity {
        commands::ProverEntity::Compile(args) => {
            let out_file_path = args.program_path.replace(".cairo", ".json");
            if let Err(err) = try_compile(&args.program_path, &out_file_path) {
                eprintln!("{}", err);
            } else {
                println!("Compiled cairo program");
            }
        }
        commands::ProverEntity::Prove(args) => {
            // verify input file is .cairo
            if args.program_path.contains(".cairo") {
                eprintln!("\nYou are trying to prove a non compiled Cairo program. Please compile it before sending it to the prover.\n");
                return;
            }

            let Some((proof, pub_inputs)) = generate_proof(&args.program_path, &proof_options)
            else {
                return;
            };

            write_proof(proof, pub_inputs, args.proof_path);
        }
        commands::ProverEntity::Verify(args) => {
            let Ok(program_content) = std::fs::read(&args.proof_path) else {
                eprintln!("Error opening {} file", args.proof_path);
                return;
            };
            let mut bytes = program_content.as_slice();
            if bytes.len() < 8 {
                eprintln!("Error reading proof from file: {}", args.proof_path);
                return;
            }

            let proof_len = usize::from_be_bytes(bytes[0..8].try_into().unwrap());
            bytes = &bytes[8..];
            if bytes.len() < proof_len {
                eprintln!("Error reading proof from file: {}", args.proof_path);
                return;
            }

            let Ok((proof, _)) = bincode::serde::decode_from_slice(
                &bytes[0..proof_len],
                bincode::config::standard(),
            ) else {
                println!("Error reading proof from file: {}", args.proof_path);
                return;
            };
            bytes = &bytes[proof_len..];

            let Ok(pub_inputs) = PublicInputs::deserialize(bytes) else {
                println!("Error reading proof from file: {}", args.proof_path);
                return;
            };

            verify_proof(proof, pub_inputs, &proof_options);
            let foo: u32 = -1;
        }
        commands::ProverEntity::ProveAndVerify(args) => {
            if args.program_path.contains(".cairo") {
                eprintln!("\nYou are trying to prove a non compiled Cairo program. Please compile it before sending it to the prover.\n");
                return;
            }

            let Some((proof, pub_inputs)) = generate_proof(&args.program_path, &proof_options)
            else {
                return;
            };
            verify_proof(proof, pub_inputs, &proof_options);
        }
        commands::ProverEntity::CompileAndProve(args) => {
            let out_file_path = args.program_path.replace(".cairo", ".json");
            match try_compile(&args.program_path, &out_file_path) {
                Ok(_) => {
                    let Some((proof, pub_inputs)) = generate_proof(&out_file_path, &proof_options)
                    else {
                        return;
                    };

                    write_proof(proof, pub_inputs, args.proof_path);
                }
                Err(err) => {
                    eprintln!("{}", err)
                }
            }
        }
        commands::ProverEntity::CompileProveAndVerify(args) => {
            let out_file_path = args.program_path.replace(".cairo", ".json");
            match try_compile(&args.program_path, &out_file_path) {
                Ok(_) => {
                    let Some((proof, pub_inputs)) = generate_proof(&out_file_path, &proof_options)
                    else {
                        return;
                    };
                    verify_proof(proof, pub_inputs, &proof_options);
                }
                Err(err) => {
                    eprintln!("{}", err)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    // use super::;

    #[test]
    fn test_field_serialization() {
        type F = lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
        type FE = stark_platinum_prover::fri::FieldElement<F>;
        let field: FE = FE::from(26);

        let serial = bincode::serde::encode_to_vec(&field, bincode::config::standard()).unwrap();

        let (field_deserial, _): (FE, _) =
            bincode::serde::decode_from_slice(&serial, bincode::config::standard()).unwrap();

        assert_eq!(field, field_deserial);
    }
}
