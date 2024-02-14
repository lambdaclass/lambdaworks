use lambdaworks_crypto::merkle_tree::proof::Proof;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
use stark_platinum_prover::{config::Commitment, fri::FieldElement, proof::stark::StarkProof};
use std::{
    collections::{BTreeSet, HashMap, HashSet},
    vec,
};

use crate::Felt252;

use super::CairoCompatibleSerializer;

#[derive(Debug, Clone)]
pub struct StarkWitness {
    pub traces_decommitment: TracesDecommitment,
    pub traces_witness: TracesWitness,
    pub composition_decommitment: TableDecommitment,
    pub composition_witness: TableCommitmentWitness,
    pub fri_witness: FriWitness,
}

#[derive(Debug, Clone)]
pub struct TracesDecommitment {
    pub original: TableDecommitment,
    pub interaction: TableDecommitment,
}

#[derive(Debug, Clone)]
pub struct TableDecommitment {
    pub n_values: Felt252,
    pub values: Vec<Felt252>,
}

#[derive(Debug, Clone)]
pub struct TracesWitness {
    pub original: TableCommitmentWitness,
    pub interaction: TableCommitmentWitness,
}

#[derive(Debug, Clone)]
pub struct TableCommitmentWitness {
    pub vector: VectorCommitmentWitness,
}

#[derive(Debug, Clone)]
pub struct VectorCommitmentWitness {
    pub n_authentications: Felt252,
    pub authentications: Proof<Commitment>,
}

#[derive(Debug, Clone)]
pub struct FriWitness {
    pub layers: Vec<FriLayerWitness>,
}

#[derive(Debug, Clone)]
pub struct FriLayerWitness {
    pub n_leaves: Felt252,
    pub leaves: Vec<Felt252>,
    pub table_witness: TableCommitmentWitness,
}

impl StarkWitness {
    pub fn convert(
        proof: &StarkProof<Stark252PrimeField, Stark252PrimeField>,
        fri_query_indexes: &[usize],
    ) -> Self {
        StarkWitness {
            traces_decommitment: TracesDecommitment::convert(proof, fri_query_indexes),
            traces_witness: TracesWitness::convert(proof, fri_query_indexes),
            composition_decommitment: TableDecommitment::convert(proof, fri_query_indexes),
            composition_witness: TableCommitmentWitness::convert(proof, fri_query_indexes),
            fri_witness: FriWitness::convert(proof, fri_query_indexes),
        }
    }
}

impl TracesDecommitment {
    pub fn convert(
        proof: &StarkProof<Stark252PrimeField, Stark252PrimeField>,
        fri_query_indexes: &[usize],
    ) -> Self {
        let mut original = TableDecommitment {
            n_values: 0.into(),
            values: vec![],
        };

        let mut interaction = TableDecommitment {
            n_values: 0.into(),
            values: vec![],
        };

        let mut fri_first_layer_openings: Vec<_> = proof
            .deep_poly_openings
            .iter()
            .zip(fri_query_indexes.iter())
            .collect();

        // Remove repeated values
        let mut seen = HashSet::new();
        fri_first_layer_openings.retain(|&(_, index)| seen.insert(index));
        // Sort by increasing value of query
        fri_first_layer_openings.sort_by(|a, b| a.1.cmp(b.1));

        // Append BT_{i_1} | BT_{i_2} | ... | BT_{i_k}
        for (opening, _) in fri_first_layer_openings.iter() {
            for elem in opening.main_trace_polys.evaluations.iter() {
                original.values.push(*elem);
            }
            if let Some(aux) = &opening.aux_trace_polys {
                for elem in aux.evaluations.iter() {
                    interaction.values.push(*elem);
                }
            }

            for elem in opening.main_trace_polys.evaluations_sym.iter() {
                original.values.push(*elem);
            }
            if let Some(aux) = &opening.aux_trace_polys {
                for elem in aux.evaluations_sym.iter() {
                    interaction.values.push(*elem);
                }
            }
        }

        original.n_values = (original.values.len() as u64).into();
        interaction.n_values = (interaction.values.len() as u64).into();

        TracesDecommitment {
            original,
            interaction,
        }
    }
}

impl TracesWitness {
    pub fn convert(
        proof: &StarkProof<Stark252PrimeField, Stark252PrimeField>,
        fri_query_indexes: &[usize],
    ) -> Self {
        let mut original = TableCommitmentWitness {
            vector: VectorCommitmentWitness {
                n_authentications: 0.into(),
                authentications: Proof {
                    merkle_path: vec![],
                },
            },
        };

        let mut interaction = TableCommitmentWitness {
            vector: VectorCommitmentWitness {
                n_authentications: 0.into(),
                authentications: Proof {
                    merkle_path: vec![],
                },
            },
        };

        let fri_trace_query_indexes: Vec<_> = fri_query_indexes
            .iter()
            .flat_map(|query| vec![query * 2, query * 2 + 1])
            .collect();

        // Append TraceMergedPaths
        // Main trace
        let fri_trace_paths: Vec<_> = proof
            .deep_poly_openings
            .iter()
            .flat_map(|opening| {
                vec![
                    &opening.main_trace_polys.proof,
                    &opening.main_trace_polys.proof_sym,
                ]
            })
            .collect();
        let nodes = CairoCompatibleSerializer::merge_authentication_paths(
            &fri_trace_paths,
            &fri_trace_query_indexes,
        );
        for node in nodes.iter() {
            original.vector.authentications.merkle_path.push(*node);
        }

        // Aux trace
        let mut all_openings_aux_trace_polys_are_some = true;
        let mut fri_trace_paths: Vec<&Proof<Commitment>> = Vec::new();
        for opening in proof.deep_poly_openings.iter() {
            if let Some(aux_trace_polys) = &opening.aux_trace_polys {
                fri_trace_paths.push(&aux_trace_polys.proof);
                fri_trace_paths.push(&aux_trace_polys.proof_sym);
            } else {
                all_openings_aux_trace_polys_are_some = false;
            }
        }
        if all_openings_aux_trace_polys_are_some {
            let nodes = CairoCompatibleSerializer::merge_authentication_paths(
                &fri_trace_paths,
                &fri_trace_query_indexes,
            );
            for node in nodes.iter() {
                interaction.vector.authentications.merkle_path.push(*node);
            }
        }

        original.vector.n_authentications =
            (original.vector.authentications.merkle_path.len() as u64).into();
        interaction.vector.n_authentications =
            (interaction.vector.authentications.merkle_path.len() as u64).into();

        TracesWitness {
            original,
            interaction,
        }
    }
}

impl TableDecommitment {
    pub fn convert(
        proof: &StarkProof<Stark252PrimeField, Stark252PrimeField>,
        fri_query_indexes: &[usize],
    ) -> Self {
        let mut commitment = TableDecommitment {
            n_values: 0.into(),
            values: vec![],
        };

        let mut fri_first_layer_openings: Vec<_> = proof
            .deep_poly_openings
            .iter()
            .zip(fri_query_indexes.iter())
            .collect();

        // Remove repeated values
        let mut seen = HashSet::new();
        fri_first_layer_openings.retain(|&(_, index)| seen.insert(index));
        // Sort by increasing value of query
        fri_first_layer_openings.sort_by(|a, b| a.1.cmp(b.1));

        // Append BH_{i_1} | BH_{i_2} | ... | B_{i_k}
        for (opening, _) in fri_first_layer_openings.iter() {
            for elem in opening.composition_poly.evaluations.iter() {
                commitment.values.push(*elem);
            }
            for elem in opening.composition_poly.evaluations_sym.iter() {
                commitment.values.push(*elem);
            }
        }

        commitment.n_values = (commitment.values.len() as u64).into();

        commitment
    }
}

impl TableCommitmentWitness {
    pub fn convert(
        proof: &StarkProof<Stark252PrimeField, Stark252PrimeField>,
        fri_query_indexes: &[usize],
    ) -> Self {
        let mut witness = TableCommitmentWitness {
            vector: VectorCommitmentWitness {
                n_authentications: 0.into(),
                authentications: Proof {
                    merkle_path: vec![],
                },
            },
        };

        // Append CompositionMergedPaths
        let fri_composition_paths: Vec<_> = proof
            .deep_poly_openings
            .iter()
            .map(|opening| &opening.composition_poly.proof)
            .collect();
        let nodes = CairoCompatibleSerializer::merge_authentication_paths(
            &fri_composition_paths,
            fri_query_indexes,
        );
        for node in nodes.iter() {
            witness.vector.authentications.merkle_path.push(*node);
        }

        witness.vector.n_authentications =
            (witness.vector.authentications.merkle_path.len() as u64).into();

        witness
    }
}

impl FriWitness {
    pub fn convert(
        proof: &StarkProof<Stark252PrimeField, Stark252PrimeField>,
        fri_query_indexes: &[usize],
    ) -> Self {
        let mut fri_witness = FriWitness { layers: vec![] };

        let mut fri_layers_evaluations: HashMap<(u64, usize, usize), FieldElement<_>> =
            HashMap::new();
        for (decommitment, query_index) in proof.query_list.iter().zip(fri_query_indexes.iter()) {
            let mut query_layer_index = *query_index;
            for (i, element) in decommitment.layers_evaluations_sym.iter().enumerate() {
                fri_layers_evaluations.insert(
                    (
                        i as u64,
                        query_layer_index >> 1,
                        (query_layer_index + 1) % 2,
                    ),
                    *element,
                );
                query_layer_index >>= 1;
            }
        }

        let mut indexes_previous_layer = fri_query_indexes.to_owned();
        for i in 0..proof.query_list[0].layers_evaluations_sym.len() {
            let mut fri_layer_witness = FriLayerWitness {
                n_leaves: 0.into(),
                leaves: vec![],
                table_witness: TableCommitmentWitness {
                    vector: VectorCommitmentWitness {
                        n_authentications: 0.into(),
                        authentications: Proof {
                            merkle_path: vec![],
                        },
                    },
                },
            };

            // Compute set Y_i
            let reconstructed_row_col: BTreeSet<_> = indexes_previous_layer
                .iter()
                .map(|index| (index >> 1, index % 2))
                .collect();

            // Compute set X_i
            let reconstructed_row_col_sym: BTreeSet<_> = reconstructed_row_col
                .iter()
                .map(|(x, y)| (*x, 1 - y))
                .collect();

            // Compute set Z_i
            let row_col_to_send: Vec<_> = reconstructed_row_col_sym
                .difference(&reconstructed_row_col)
                .collect();

            // Append Z_i
            for element in row_col_to_send
                .iter()
                .map(|(row, col)| &fri_layers_evaluations[&(i as u64, *row, *col)])
            {
                fri_layer_witness.leaves.push(*element);
            }

            fri_layer_witness.n_leaves = (fri_layer_witness.leaves.len() as u64).into();

            indexes_previous_layer = indexes_previous_layer
                .iter()
                .map(|index| index >> 1)
                .collect();

            let layer_auth_paths: Vec<_> = proof
                .query_list
                .iter()
                .map(|decommitment| &decommitment.layers_auth_paths[i])
                .collect();

            // Append MergedPathsLayer_i
            let nodes = CairoCompatibleSerializer::merge_authentication_paths(
                &layer_auth_paths,
                &indexes_previous_layer,
            );
            for node in nodes.iter() {
                fri_layer_witness
                    .table_witness
                    .vector
                    .authentications
                    .merkle_path
                    .push(*node);
            }

            fri_layer_witness.table_witness.vector.n_authentications = (fri_layer_witness
                .table_witness
                .vector
                .authentications
                .merkle_path
                .len()
                as u64)
                .into();
            fri_witness.layers.push(fri_layer_witness);
        }

        fri_witness
    }
}
