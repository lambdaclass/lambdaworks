#![no_main]
use libfuzzer_sys::fuzz_target;
use lambdaworks_crypto::hash::poseidon::starknet::PoseidonCairoStark252;
use lambdaworks_crypto::hash::poseidon::Poseidon;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
use lambdaworks_math::traits::ByteConversion;
use pathfinder_crypto::MontFelt;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;

    /*fft::{
        gpu::metal::ops::fft as fft_metal,
        cpu::{
            roots_of_unity::get_twiddles,
            ops::fft as fft_cpu
        }
    },
    field::{
        traits::RootsConfig,
        fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
        element::FieldElement
    },
};
use lambdaworks_gpu::metal::abstractions::state::MetalState;
 */

// Fuzz target for Poseidon hash function

fn compare_errors()

fuzz_target!(|data: &[u8]| {
    // Asegúrate de que los datos de entrada tengan suficiente longitud para dividirlos en dos partes.
    // Esta verificación previene los pánicos por índices fuera de límites.
    if data.len() >= 64 {  // Garantiza al menos 32 bytes para cada entrada.
        // Divide los datos de entrada en dos partes para simular dos entradas para la función hash.
        let (input1, input2) = data.split_at(32);  // Divide en dos partes de 32 bytes.

        // Intenta convertir las partes de entrada en FieldElement de Stark252PrimeField.
        if let (Ok(felt1), Ok(felt2)) = (
            FieldElement::<Stark252PrimeField>::from_bytes_be(input1),
            FieldElement::<Stark252PrimeField>::from_bytes_be(input2),
        ) {
            // Ejecuta la función hash Poseidon con los FieldElements generados.
            let _hash_result = poseidon_hash(&felt1, &felt2);

            // No es necesario validar el `_hash_result` en la mayoría de los escenarios de fuzzing,
            // ya que el enfoque está en encontrar comportamientos inesperados, como pánicos o errores.
        }
    }
});

