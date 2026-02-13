//! Poseidon2 parameters for Goldilocks field (p = 2^64 - 2^32 + 1).
//!
//! # Source
//!
//! Constants derived from HorizenLabs/Plonky3 Poseidon2 implementation.
//! - Repository: <https://github.com/HorizenLabs/poseidon2>
//! - Plonky3 reference: <https://github.com/Plonky3/Plonky3>
//!
//! The round constants and diagonal matrix values were generated using the
//! Poseidon2 parameter generation algorithm with Goldilocks-specific security
//! margins.
//!
//! # Configuration (WIDTH=8)
//!
//! - S-box: x^7 (gcd(7, p-1) = 1, ensuring bijection)
//! - External rounds: 4 (initial) + 4 (terminal)
//! - Internal rounds: 22
//! - Rate: 4
//! - Capacity: 4
//! - Security: ~128-bit
//!
//! All constants are stored as precomputed `Fp` values using `const_from_raw`
//! to avoid runtime reduction (all values are already canonical, i.e. < p).

use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;

/// Type alias for Goldilocks field element (local to constants module)
type Fp = FieldElement<Goldilocks64Field>;

/// Create an Fp constant at compile time, skipping runtime modular reduction.
/// Safety: all values in this file are canonical (< p = 2^64 - 2^32 + 1).
const fn fp(value: u64) -> Fp {
    Fp::const_from_raw(value)
}

/// Width of the permutation state
pub const WIDTH: usize = 8;

/// Number of external (full) rounds at the beginning
pub const EXTERNAL_ROUNDS_BEGIN: usize = 4;

/// Number of external (full) rounds at the end
pub const EXTERNAL_ROUNDS_END: usize = 4;

/// Number of internal (partial) rounds
pub const INTERNAL_ROUNDS: usize = 22;

/// Rate for sponge construction (absorb this many elements per permutation)
pub const RATE: usize = 4;

/// Diagonal elements for the internal diffusion matrix (width 8)
/// The internal matrix is: diag(MATRIX_DIAG) with 1s on off-diagonal
pub const MATRIX_DIAG_8: [Fp; 8] = [
    fp(0xa98811a1fed4e3a5),
    fp(0x1cc48b54f377e2a0),
    fp(0xe40cd4f6c5609a26),
    fp(0x11de79ebca97a4a3),
    fp(0x9177c73d8b7e929c),
    fp(0x2a6fe8085797e791),
    fp(0x3de6e93329f8d5ad),
    fp(0x3f7af9125da962fe),
];

/// External round constants - initial 4 rounds (4 rounds x 8 values)
pub const EXTERNAL_ROUND_CONSTANTS_INIT: [[Fp; 8]; 4] = [
    [
        fp(0xdd5743e7f2a5a5d9),
        fp(0xcb3a864e58ada44b),
        fp(0xffa2449ed32f8cdc),
        fp(0x42025f65d6bd13ee),
        fp(0x7889175e25506323),
        fp(0x34b98bb03d24b737),
        fp(0xbdcc535ecc4faa2a),
        fp(0x5b20ad869fc0d033),
    ],
    [
        fp(0xf1dda5b9259dfcb4),
        fp(0x27515210be112d59),
        fp(0x4227d1718c766c3f),
        fp(0x26d333161a5bd794),
        fp(0x49b938957bf4b026),
        fp(0x4a56b5938b213669),
        fp(0x1120426b48c8353d),
        fp(0x6b323c3f10a56cad),
    ],
    [
        fp(0xce57d6245ddca6b2),
        fp(0xb1fc8d402bba1eb1),
        fp(0xb5c5096ca959bd04),
        fp(0x6db55cd306d31f7f),
        fp(0xc49d293a81cb9641),
        fp(0x1ce55a4fe979719f),
        fp(0xa92e60a9d178a4d1),
        fp(0x002cc64973bcfd8c),
    ],
    [
        fp(0xcea721cce82fb11b),
        fp(0xe5b55eb8098ece81),
        fp(0x4e30525c6f1ddd66),
        fp(0x43c6702827070987),
        fp(0xaca68430a7b5762a),
        fp(0x3674238634df9c93),
        fp(0x88cee1c825e33433),
        fp(0xde99ae8d74b57176),
    ],
];

/// External round constants - terminal 4 rounds (4 rounds x 8 values)
pub const EXTERNAL_ROUND_CONSTANTS_TERM: [[Fp; 8]; 4] = [
    [
        fp(0x014ef1197d341346),
        fp(0x9725e20825d07394),
        fp(0xfdb25aef2c5bae3b),
        fp(0xbe5402dc598c971e),
        fp(0x93a5711f04cdca3d),
        fp(0xc45a9a5b2f8fb97b),
        fp(0xfe8946a924933545),
        fp(0x2af997a27369091c),
    ],
    [
        fp(0xaa62c88e0b294011),
        fp(0x058eb9d810ce9f74),
        fp(0xb3cb23eced349ae4),
        fp(0xa3648177a77b4a84),
        fp(0x43153d905992d95d),
        fp(0xf4e2a97cda44aa4b),
        fp(0x5baa2702b908682f),
        fp(0x082923bdf4f750d1),
    ],
    [
        fp(0x98ae09a325893803),
        fp(0xf8a6475077968838),
        fp(0xceb0735bf00b2c5f),
        fp(0x0a1a5d953888e072),
        fp(0x2fcb190489f94475),
        fp(0xb5be06270dec69fc),
        fp(0x739cb934b09acf8b),
        fp(0x537750b75ec7f25b),
    ],
    [
        fp(0xe9dd318bae1f3961),
        fp(0xf7462137299efe1a),
        fp(0xb1f6b8eee9adb940),
        fp(0xbdebcc8a809dfe6b),
        fp(0x40fc1f791b178113),
        fp(0x3ac1c3362d014864),
        fp(0x9a016184bdb8aeba),
        fp(0x95f2394459fbc25e),
    ],
];

/// Internal round constants (22 rounds x 1 value, applied to first element only)
pub const INTERNAL_ROUND_CONSTANTS: [Fp; 22] = [
    fp(0x488897d85ff51f56),
    fp(0x1140737ccb162218),
    fp(0xa7eeb9215866ed35),
    fp(0x9bd2976fee49fcc9),
    fp(0xc0c8f0de580a3fcc),
    fp(0x4fb2dae6ee8fc793),
    fp(0x343a89f35f37395b),
    fp(0x223b525a77ca72c8),
    fp(0x56ccb62574aaa918),
    fp(0xc4d507d8027af9ed),
    fp(0xa080673cf0b7e95c),
    fp(0xf0184884eb70dcf8),
    fp(0x044f10b0cb3d5c69),
    fp(0xe9e3f7993938f186),
    fp(0x1b761c80e772f459),
    fp(0x606cec607a1b5fac),
    fp(0x14a0c2e1d45f03cd),
    fp(0x4eace8855398574f),
    fp(0xf905ca7103eff3e6),
    fp(0xf8c8f8d20862c059),
    fp(0xb524fe8bdd678e5a),
    fp(0xfbb7865901a1ec41),
];
