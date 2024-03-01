use std::fs::File;
use std::{io, ops::Neg};

use baby_snark::{
    common::FrElement, scs::SquareConstraintSystem, setup, ssp::SquareSpanProgram, verify, Prover,
};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct Solution {
    u: Vec<Vec<i64>>,
    public: Vec<i64>,
    witness: Vec<i64>,
}
use clap::Parser;

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long)]
    file: String,
}

fn main() {
    let args = Args::parse();
    let file = File::open(args.file).unwrap();
    let reader = io::BufReader::new(file);

    let sol: Solution = serde_json::from_reader(reader).unwrap();

    let u = i64_matrix_to_field(sol.u);
    let public = i64_vec_to_field(sol.public);
    let witness = i64_vec_to_field(sol.witness);

    println!("Result: {}", test_integration(u, witness, public));
}

fn test_integration(
    u: Vec<Vec<FrElement>>,
    witness: Vec<FrElement>,
    public: Vec<FrElement>,
) -> bool {
    let mut input = public.clone();
    input.extend(witness.clone());

    let ssp = SquareSpanProgram::from_scs(SquareConstraintSystem::from_matrix(u, public.len()));
    let (proving_key, verifying_key) = setup(&ssp);

    let proof = Prover::prove(&input, &ssp, &proving_key);
    verify(&verifying_key, &proof, &public)
}

fn i64_to_field(element: &i64) -> FrElement {
    let mut fr_element = FrElement::from(element.unsigned_abs());
    if element.is_negative() {
        fr_element = fr_element.neg()
    }

    fr_element
}

fn i64_vec_to_field(elements: Vec<i64>) -> Vec<FrElement> {
    elements.iter().map(i64_to_field).collect()
}

fn i64_matrix_to_field(elements: Vec<Vec<i64>>) -> Vec<Vec<FrElement>> {
    let mut matrix = Vec::new();
    for f in elements {
        matrix.push(i64_vec_to_field(f));
    }
    matrix
}
