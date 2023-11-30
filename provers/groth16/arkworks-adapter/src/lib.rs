#[cfg(test)]
mod integration_tests;

use ark_ff::PrimeField;
use ark_relations::r1cs::{ConstraintSystemRef, Field};
use lambdaworks_groth16::{common::*, r1cs::R1CS};
use lambdaworks_math::traits::ByteConversion;

use std::ops::Deref;

pub fn r1cs_from_arkworks_cs<F: Field>(cs: &ConstraintSystemRef<F>) -> R1CS {
    cs.inline_all_lcs();

    let r1cs_matrices = cs.to_matrices().unwrap();
    let total_variables = cs.num_witness_variables() + cs.num_instance_variables() - 1;

    let a = arkworks_r1cs_matrix_to_qap(&r1cs_matrices.a, total_variables);
    let b = arkworks_r1cs_matrix_to_qap(&r1cs_matrices.b, total_variables);
    let c = arkworks_r1cs_matrix_to_qap(&r1cs_matrices.c, total_variables);

    a.iter().for_each(|row| {
        row.iter().for_each(|e| {
            print!("{}\t", e.to_string());
        });
        println!();
    });
    println!("================================");
    b.iter().for_each(|row| {
        row.iter().for_each(|e| {
            print!("{}\t", e.to_string());
        });
        println!();
    });
    println!("================================");
    c.iter().for_each(|row| {
        row.iter().for_each(|e| {
            print!("{}\t", e.to_string());
        });
        println!();
    });

    /*
        Notice we can't differentiate outputs and inputs from Arkworks CS, but for the proving system everything that matters is that it's public data (IO),
        or private data (witness/c_mid)
    */
    R1CS::new_with_matrixes(a, b, c, cs.num_instance_variables() - 1, 0)
}

pub fn io_and_witness_from_arkworks_cs<F: PrimeField>(
    cs: &ConstraintSystemRef<F>,
) -> Vec<FrElement> {
    let binding = cs.borrow().unwrap();
    let borrowed_cs_ref = binding.deref();

    let ark_witness = &borrowed_cs_ref.witness_assignment;
    let ark_io = &borrowed_cs_ref.instance_assignment[1..].to_vec();

    let mut witness = vec![FrElement::one()];
    witness.extend(ark_io.iter().map(ark_fr_to_fr_element));
    witness.extend(ark_witness.iter().map(ark_fr_to_fr_element));

    witness
}

fn arkworks_r1cs_matrix_to_qap<F: Field>(
    m: &[Vec<(F, usize)>],
    total_variables: usize,
) -> Vec<Vec<FrElement>> {
    println!("======== total variables: {}", total_variables);
    m.iter().for_each(|row| println!("[{:?}]", row));

    sparse_matrix_to_dense(&arkworks_matrix_fps_to_fr_elements(m), total_variables)
}

fn arkworks_matrix_fps_to_fr_elements<F: Field>(
    m: &[Vec<(F, usize)>],
) -> Vec<Vec<(FrElement, usize)>> {
    m.iter()
        .map(|x| {
            x.iter()
                .map(|(x, y)| (ark_fr_to_fr_element(x), *y))
                .collect()
        })
        .collect()
}

fn ark_fr_to_fr_element<F: Field>(ark_fq: &F) -> FrElement {
    let mut buff = Vec::<u8>::new();
    ark_fq.serialize_compressed(&mut buff).unwrap();
    FrElement::from_bytes_le(&buff).unwrap()
}

fn sparse_matrix_to_dense(
    m: &[Vec<(FrElement, usize)>],
    total_variables: usize,
) -> Vec<Vec<FrElement>> {
    m.iter()
        .map(|row| sparse_row_to_dense(row, total_variables))
        .collect()
}

fn sparse_row_to_dense(row: &Vec<(FrElement, usize)>, total_variables: usize) -> Vec<FrElement> {
    //The first column of the r1cs is used for constants

    let mut dense_row = vec![FrElement::from(0); total_variables + 1];

    for element in row {
        dense_row[element.1] = element.0.clone();
    }
    dense_row
}
