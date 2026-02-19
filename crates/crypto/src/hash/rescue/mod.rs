mod parameters;
mod rpo;
mod rpx;
mod utils;

pub use parameters::SecurityLevel;
pub use rpo::MdsMethod;
pub use rpo::Rpo256;
pub use rpx::Rpx256;

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;

pub type Fp = FieldElement<Goldilocks64Field>;
