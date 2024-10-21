mod mds_method;
mod parameters;
mod rescue_prime_optimized;
mod utils;

pub use mds_method::MdsMethod;
pub use rescue_prime_optimized::RescuePrimeOptimized;

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;

pub type Fp = FieldElement<Goldilocks64Field>;
