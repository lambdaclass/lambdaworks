// =============================================================================
// Poseidon hash for the Goldilocks field — Metal GPU implementation.
//
// Matches the CPU implementation in crypto/hash/poseidon/goldilocks/.
// Parameters: width=12, rate=8, capacity=4, x^7 S-box,
//             4+4 full rounds, 22 partial rounds.
// MDS matrix: circulant + diagonal.
// Round constants sourced from Plonky2.
//
// Uses fast partial round optimization (Plonky2 technique) that reduces
// partial round MDS from O(n^2) to O(n) multiplications per round.
//
// NOTE: This shader includes Goldilocks field arithmetic inline (from fp_u64.h.metal)
// because it is loaded at runtime via include_str!().
// =============================================================================

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Goldilocks field arithmetic (duplicated from fp_u64.h.metal for standalone use)
// =============================================================================

constant uint64_t GL_EPSILON = 0xFFFFFFFF;
constant uint64_t GL_PRIME   = 0xFFFFFFFF00000001UL;

// Field addition: (a + b) mod p
inline uint64_t gl_add(uint64_t a, uint64_t b) {
    uint64_t sum = a + b;
    bool over = sum < a;
    uint64_t sum2 = sum + (over ? GL_EPSILON : 0);
    bool over2 = over && (sum2 < sum);
    return sum2 + (over2 ? GL_EPSILON : 0);
}

// Field subtraction: (a - b) mod p
inline uint64_t gl_sub(uint64_t a, uint64_t b) {
    uint64_t diff = a - b;
    bool under = a < b;
    uint64_t diff2 = diff - (under ? GL_EPSILON : 0);
    bool under2 = under && (diff2 > diff);
    return diff2 - (under2 ? GL_EPSILON : 0);
}

// Field multiplication: (a * b) mod p using 128-bit intermediate
inline uint64_t gl_mul(uint64_t a, uint64_t b) {
    uint64_t lo = a * b;
    uint64_t hi = metal::mulhi(a, b);

    // Reduce 128-bit product using 2^64 ≡ 2^32 - 1 (mod p)
    uint64_t hi_hi = hi >> 32;
    uint64_t hi_lo = hi & GL_EPSILON;

    // t0 = lo - hi_hi
    uint64_t t0 = lo - hi_hi;
    bool borrow = lo < hi_hi;
    t0 = borrow ? t0 - GL_EPSILON : t0;

    // t1 = hi_lo * EPSILON = (hi_lo << 32) - hi_lo
    uint64_t t1 = (hi_lo << 32) - hi_lo;

    // result = t0 + t1
    uint64_t result = t0 + t1;
    bool carry = result < t0;
    result = carry ? result + GL_EPSILON : result;

    return result;
}

// Canonicalize to [0, p)
inline uint64_t gl_canon(uint64_t a) {
    return a >= GL_PRIME ? a - GL_PRIME : a;
}

// Reduce (hi:lo) mod Goldilocks
inline uint64_t gl_reduce_u128(uint64_t lo, uint64_t hi) {
    uint64_t hi_hi = hi >> 32;
    uint64_t hi_lo = hi & GL_EPSILON;

    uint64_t t0 = lo - hi_hi;
    bool borrow = lo < hi_hi;
    t0 = borrow ? t0 - GL_EPSILON : t0;

    uint64_t t1 = (hi_lo << 32) - hi_lo;

    uint64_t result = t0 + t1;
    bool carry = result < t0;
    result = carry ? result + GL_EPSILON : result;

    return gl_canon(result);
}

// Accumulate a*b into (lo, hi) with carry tracking
inline void acc_mul(thread uint64_t& lo, thread uint64_t& hi, uint64_t a, uint64_t b) {
    uint64_t prod_lo = a * b;
    uint64_t prod_hi = metal::mulhi(a, b);
    uint64_t old_lo = lo;
    lo += prod_lo;
    if (lo < old_lo) hi++;
    hi += prod_hi;
}

// =============================================================================
// Poseidon constants
// =============================================================================

constant uint SPONGE_WIDTH       = 12;
constant uint SPONGE_RATE        = 8;
constant uint HALF_N_FULL_ROUNDS = 4;
constant uint N_PARTIAL_ROUNDS   = 22;
constant uint N_ROUNDS           = 30; // 8 full + 22 partial

// MDS matrix: circ(MDS_CIRC) + diag(MDS_DIAG)
constant uint64_t MDS_CIRC[12] = { 17, 15, 41, 16, 2, 28, 13, 13, 39, 18, 34, 20 };
constant uint64_t MDS_DIAG[12] = {  8,  0,  0,  0, 0,  0,  0,  0,  0,  0,  0,  0 };

// Round constants for full rounds only (rounds 0-3 and 26-29)
constant uint64_t RC[360] = {
    // Round 0 (full)
    0xb585f766f2144405UL, 0x7746a55f43921ad7UL, 0xb2fb0d31cee799b4UL, 0x0f6760a4803427d7UL,
    0xe10d666650f4e012UL, 0x8cae14cb07d09bf1UL, 0xd438539c95f63e9fUL, 0xef781c7ce35b4c3dUL,
    0xcdc4a239b0c44426UL, 0x277fa208bf337bffUL, 0xe17653a29da578a1UL, 0xc54302f225db2c76UL,
    // Round 1 (full)
    0x86287821f722c881UL, 0x59cd1a8a41c18e55UL, 0xc3b919ad495dc574UL, 0xa484c4c5ef6a0781UL,
    0x308bbd23dc5416ccUL, 0x6e4a40c18f30c09cUL, 0x9a2eedb70d8f8cfaUL, 0xe360c6e0ae486f38UL,
    0xd5c7718fbfc647fbUL, 0xc35eae071903ff0bUL, 0x849c2656969c4be7UL, 0xc0572c8c08cbbbadUL,
    // Round 2 (full)
    0xe9fa634a21de0082UL, 0xf56f6d48959a600dUL, 0xf7d713e806391165UL, 0x8297132b32825dafUL,
    0xad6805e0e30b2c8aUL, 0xac51d9f5fcf8535eUL, 0x502ad7dc18c2ad87UL, 0x57a1550c110b3041UL,
    0x66bbd30e6ce0e583UL, 0x0da2abef589d644eUL, 0xf061274fdb150d61UL, 0x28b8ec3ae9c29633UL,
    // Round 3 (full)
    0x92a756e67e2b9413UL, 0x70e741ebfee96586UL, 0x019d5ee2af82ec1cUL, 0x6f6f2ed772466352UL,
    0x7cf416cfe7e14ca1UL, 0x61df517b86a46439UL, 0x85dc499b11d77b75UL, 0x4b959b48b9c10733UL,
    0xe8be3e5da8043e57UL, 0xf5c0bc1de6da8699UL, 0x40b12cbf09ef74bfUL, 0xa637093ecb2ad631UL,
    // Rounds 4-25 (partial — kept for reference; fast partial rounds use FP_* constants)
    0x3cc3f892184df408UL, 0x2e479dc157bf31bbUL, 0x6f49de07a6234346UL, 0x213ce7bede378d7bUL,
    0x5b0431345d4dea83UL, 0xa2de45780344d6a1UL, 0x7103aaf94a7bf308UL, 0x5326fc0d97279301UL,
    0xa9ceb74fec024747UL, 0x27f8ec88bb21b1a3UL, 0xfceb4fda1ded0893UL, 0xfac6ff1346a41675UL,
    0x7131aa45268d7d8cUL, 0x9351036095630f9fUL, 0xad535b24afc26bfbUL, 0x4627f5c6993e44beUL,
    0x645cf794b8f1cc58UL, 0x241c70ed0af61617UL, 0xacb8e076647905f1UL, 0x3737e9db4c4f474dUL,
    0xe7ea5e33e75fffb6UL, 0x90dee49fc9bfc23aUL, 0xd1b1edf76bc09c92UL, 0x0b65481ba645c602UL,
    0x99ad1aab0814283bUL, 0x438a7c91d416ca4dUL, 0xb60de3bcc5ea751cUL, 0xc99cab6aef6f58bcUL,
    0x69a5ed92a72ee4ffUL, 0x5e7b329c1ed4ad71UL, 0x5fc0ac0800144885UL, 0x32db829239774ecaUL,
    0x0ade699c5830f310UL, 0x7cc5583b10415f21UL, 0x85df9ed2e166d64fUL, 0x6604df4fee32bcb1UL,
    0xeb84f608da56ef48UL, 0xda608834c40e603dUL, 0x8f97fe408061f183UL, 0xa93f485c96f37b89UL,
    0x6704e8ee8f18d563UL, 0xcee3e9ac1e072119UL, 0x510d0e65e2b470c1UL, 0xf6323f486b9038f0UL,
    0x0b508cdeffa5ceefUL, 0xf2417089e4fb3cbdUL, 0x60e75c2890d15730UL, 0xa6217d8bf660f29cUL,
    0x7159cd30c3ac118eUL, 0x839b4e8fafead540UL, 0x0d3f3e5e82920adcUL, 0x8f7d83bddee7bba8UL,
    0x780f2243ea071d06UL, 0xeb915845f3de1634UL, 0xd19e120d26b6f386UL, 0x016ee53a7e5fecc6UL,
    0xcb5fd54e7933e477UL, 0xacb8417879fd449fUL, 0x9c22190be7f74732UL, 0x5d693c1ba3ba3621UL,
    0xdcef0797c2b69ec7UL, 0x3d639263da827b13UL, 0xe273fd971bc8d0e7UL, 0x418f02702d227ed5UL,
    0x8c25fda3b503038cUL, 0x2cbaed4daec8c07cUL, 0x5f58e6afcdd6ddc2UL, 0x284650ac5e1b0ebaUL,
    0x635b337ee819dab5UL, 0x9f9a036ed4f2d49fUL, 0xb93e260cae5c170eUL, 0xb0a7eae879ddb76dUL,
    0xd0762cbc8ca6570cUL, 0x34c6efb812b04bf5UL, 0x40bf0ab5fa14c112UL, 0xb6b570fc7c5740d3UL,
    0x5a27b9002de33454UL, 0xb1a5b165b6d2b2d2UL, 0x8722e0ace9d1be22UL, 0x788ee3b37e5680fbUL,
    0x14a726661551e284UL, 0x98b7672f9ef3b419UL, 0xbb93ae776bb30e3aUL, 0x28fd3b046380f850UL,
    0x30a4680593258387UL, 0x337dc00c61bd9ce1UL, 0xd5eca244c7a4ff1dUL, 0x7762638264d279bdUL,
    0xc1e434bedeefd767UL, 0x0299351a53b8ec22UL, 0xb2d456e4ad251b80UL, 0x3e9ed1fda49cea0bUL,
    0x2972a92ba450bed8UL, 0x20216dd77be493deUL, 0xadffe8cf28449ec6UL, 0x1c4dbb1c4c27d243UL,
    0x15a16a8a8322d458UL, 0x388a128b7fd9a609UL, 0x2300e5d6baedf0fbUL, 0x2f63aa8647e15104UL,
    0xf1c36ce86ecec269UL, 0x27181125183970c9UL, 0xe584029370dca96dUL, 0x4d9bbc3e02f1cfb2UL,
    0xea35bc29692af6f8UL, 0x18e21b4beabb4137UL, 0x1e3b9fc625b554f4UL, 0x25d64362697828fdUL,
    0x5a3f1bb1c53a9645UL, 0xdb7f023869fb8d38UL, 0xb462065911d4e1fcUL, 0x49c24ae4437d8030UL,
    0xd793862c112b0566UL, 0xaadd1106730d8febUL, 0xc43b6e0e97b0d568UL, 0xe29024c18ee6fca2UL,
    0x5e50c27535b88c66UL, 0x10383f20a4ff9a87UL, 0x38e8ee9d71a45af8UL, 0xdd5118375bf1a9b9UL,
    0x775005982d74d7f7UL, 0x86ab99b4dde6c8b0UL, 0xb1204f603f51c080UL, 0xef61ac8470250ecfUL,
    0x1bbcd90f132c603fUL, 0x0cd1dabd964db557UL, 0x11a3ae5beb9d1ec9UL, 0xf755bfeea585d11dUL,
    0xa3b83250268ea4d7UL, 0x516306f4927c93afUL, 0xddb4ac49c9efa1daUL, 0x64bb6dec369d4418UL,
    0xf9cc95c22b4c1fccUL, 0x08d37f755f4ae9f6UL, 0xeec49b613478675bUL, 0xf143933aed25e0b0UL,
    0xe4c5dd8255dfc622UL, 0xe7ad7756f193198eUL, 0x92c2318b87fff9cbUL, 0x739c25f8fd73596dUL,
    0x5636cac9f16dfed0UL, 0xdd8f909a938e0172UL, 0xc6401fe115063f5bUL, 0x8ad97b33f1ac1455UL,
    0x0c49366bb25e8513UL, 0x0784d3d2f1698309UL, 0x530fb67ea1809a81UL, 0x410492299bb01f49UL,
    0x139542347424b9acUL, 0x9cb0bd5ea1a1115eUL, 0x02e3f615c38f49a1UL, 0x985d4f4a9c5291efUL,
    0x775b9feafdcd26e7UL, 0x304265a6384f0f2dUL, 0x593664c39773012cUL, 0x4f0a2e5fb028f2ceUL,
    0xdd611f1000c17442UL, 0xd8185f9adfea4fd0UL, 0xef87139ca9a3ab1eUL, 0x3ba71336c34ee133UL,
    0x7d3a455d56b70238UL, 0x660d32e130182684UL, 0x297a863f48cd1f43UL, 0x90e0a736a751ebb7UL,
    0x549f80ce550c4fd3UL, 0x0f73b2922f38bd64UL, 0x16bf1f73fb7a9c3fUL, 0x6d1f5a59005bec17UL,
    0x02ff876fa5ef97c4UL, 0xc5cb72a2a51159b0UL, 0x8470f39d2d5c900eUL, 0x25abb3f1d39fcb76UL,
    0x23eb8cc9b372442fUL, 0xd687ba55c64f6364UL, 0xda8d9e90fd8ff158UL, 0xe3cbdc7d2fe45ea7UL,
    0xb9a8c9b3aee52297UL, 0xc0d28a5c10960bd3UL, 0x45d7ac9b68f71a34UL, 0xeeb76e397069e804UL,
    0x3d06c8bd1514e2d9UL, 0x9c9c98207cb10767UL, 0x65700b51aedfb5efUL, 0x911f451539869408UL,
    0x7ae6849fbc3a0ec6UL, 0x3bb340eba06afe7eUL, 0xb46e9d8b682ea65eUL, 0x8dcf22f9a3b34356UL,
    0x77bdaeda586257a7UL, 0xf19e400a5104d20dUL, 0xc368a348e46d950fUL, 0x9ef1cd60e679f284UL,
    0xe89cd854d5d01d33UL, 0x5cd377dc8bb882a2UL, 0xa7b0fb7883eee860UL, 0x7684403ec392950dUL,
    0x5fa3f06f4fed3b52UL, 0x8df57ac11bc04831UL, 0x2db01efa1e1e1897UL, 0x54846de4aadb9ca2UL,
    0xba6745385893c784UL, 0x541d496344d2c75bUL, 0xe909678474e687feUL, 0xdfe89923f6c9c2ffUL,
    0xece5a71e0cfedc75UL, 0x5ff98fd5d51fe610UL, 0x83e8941918964615UL, 0x5922040b47f150c1UL,
    0xf97d750e3dd94521UL, 0x5080d4c2b86f56d7UL, 0xa7de115b56c78d70UL, 0x6a9242ac87538194UL,
    0xf7856ef7f9173e44UL, 0x2265fc92feb0dc09UL, 0x17dfc8e4f7ba8a57UL, 0x9001a64209f21db8UL,
    0x90004c1371b893c5UL, 0xb932b7cf752e5545UL, 0xa0b1df81b6fe59fcUL, 0x8ef1dd26770af2c2UL,
    0x0541a4f9cfbeed35UL, 0x9e61106178bfc530UL, 0xb3767e80935d8af2UL, 0x0098d5782065af06UL,
    0x31d191cd5c1466c7UL, 0x410fefafa319ac9dUL, 0xbdf8f242e316c4abUL, 0x9e8cd55b57637ed0UL,
    0xde122bebe9a39368UL, 0x4d001fd58f002526UL, 0xca6637000eb4a9f8UL, 0x2f2339d624f91f78UL,
    0x6d1a7918c80df518UL, 0xdf9a4939342308e9UL, 0xebc2151ee6c8398cUL, 0x03cc2ba8a1116515UL,
    0xd341d037e840cf83UL, 0x387cb5d25af4afccUL, 0xbba2515f22909e87UL, 0x7248fe7705f38e47UL,
    0x4d61e56a525d225aUL, 0x262e963c8da05d3dUL, 0x59e89b094d220ec2UL, 0x055d5b52b78b9c5eUL,
    0x82b27eb33514ef99UL, 0xd30094ca96b7ce7bUL, 0xcf5cb381cd0a1535UL, 0xfeed4db6919e5a7cUL,
    0x41703f53753be59fUL, 0x5eeea940fcde8b6fUL, 0x4cd1f1b175100206UL, 0x4a20358574454ec0UL,
    0x1478d361dbbf9facUL, 0x6f02dc07d141875cUL, 0x296a202ed8e556a2UL, 0x2afd67999bf32ee5UL,
    0x7acfd96efa95491dUL, 0x6798ba0c0abb2c6dUL, 0x34c6f57b26c92122UL, 0x5736e1bad206b5deUL,
    0x20057d2a0056521bUL, 0x3dea5bd5d0578bd7UL, 0x16e50d897d4634acUL, 0x29bff3ecb9b7a6e3UL,
    // Rounds 26-29 (full, indices 312-359)
    0x475cd3205a3bdcdeUL, 0x18a42105c31b7e88UL, 0x023e7414af663068UL, 0x15147108121967d7UL,
    0xe4a3dff1d7d6fef9UL, 0x01a8d1a588085737UL, 0x11b4c74eda62beefUL, 0xe587cc0d69a73346UL,
    0x1ff7327017aa2a6eUL, 0x594e29c42473d06bUL, 0xf6f31db1899b12d5UL, 0xc02ac5e47312d3caUL,
    0xe70201e960cb78b8UL, 0x6f90ff3b6a65f108UL, 0x42747a7245e7fa84UL, 0xd1f507e43ab749b2UL,
    0x1c86d265f15750cdUL, 0x3996ce73dd832c1cUL, 0x8e7fba02983224bdUL, 0xba0dec7103255dd4UL,
    0x9e9cbd781628fc5bUL, 0xdae8645996edd6a5UL, 0xdebe0853b1a1d378UL, 0xa49229d24d014343UL,
    0x7be5b9ffda905e1cUL, 0xa3c95eaec244aa30UL, 0x0230bca8f4df0544UL, 0x4135c2bebfe148c6UL,
    0x166fc0cc438a3c72UL, 0x3762b59a8ae83efaUL, 0xe8928a4c89114750UL, 0x2a440b51a4945ee5UL,
    0x80cefd2b7d99ff83UL, 0xbb9879c6e61fd62aUL, 0x6e7c8f1a84265034UL, 0x164bb2de1bbeddc8UL,
    0xf3c12fe54d5c653bUL, 0x40b9e922ed9771e2UL, 0x551f5b0fbe7b1840UL, 0x25032aa7c4cb1811UL,
    0xaaed34074b164346UL, 0x8ffd96bbf9c9c81dUL, 0x70fc91eb5937085cUL, 0x7f795e2a5f915440UL,
    0x4543d9df5476d3cbUL, 0xf172d73e004fc90dUL, 0xdfd1c4febcc81238UL, 0xbc8dfb627fe558fcUL,
};

// =============================================================================
// Fast partial round constants (from Plonky2)
// Reduces partial round MDS from O(n^2) to O(n) per round.
// =============================================================================

// Modified constants for the first partial round (applied to all 12 state elements)
constant uint64_t FP_FIRST_RC[12] = {
    0x3cc3f892184df408UL, 0xe993fd841e7e97f1UL, 0xf2831d3575f0f3afUL, 0xd2500e0a350994caUL,
    0xc5571f35d7288633UL, 0x91d89c5184109a02UL, 0xf37f925d04e5667bUL, 0x2d6e448371955a69UL,
    0x740ef19ce01398a1UL, 0x694d24c0752fdf45UL, 0x60936af96ee2f148UL, 0xc33448feadc78f0cUL,
};

// Per-partial-round scalar constant (added to state[0] only, 22 values)
constant uint64_t FP_RC[22] = {
    0x74cb2e819ae421abUL, 0xd2559d2370e7f663UL, 0x62bf78acf843d17cUL, 0xd5ab7b67e14d1fb4UL,
    0xb9fe2ae6e0969bdcUL, 0xe33fdf79f92a10e8UL, 0x0ea2bb4c2b25989bUL, 0xca9121fbf9d38f06UL,
    0xbdd9b0aa81f58fa4UL, 0x83079fa4ecf20d7eUL, 0x650b838edfcc4ad3UL, 0x77180c88583c76acUL,
    0xaf8c20753143a180UL, 0xb8ccfe9989a39175UL, 0x954a1729f60cc9c5UL, 0xdeb5b550c4dca53bUL,
    0xf01bb0b00f77011eUL, 0xa1ebb404b676afd9UL, 0x860b6e1597a0173eUL, 0x308bb65a036acbceUL,
    0x1aca78f31c97c876UL, 0x0UL,
};

// W_hat vectors for computing result[0] in each partial round (22 × 11, flattened)
constant uint64_t FP_W_HATS[242] = {
    // Round 0
    0x3d999c961b7c63b0UL, 0x814e82efcd172529UL, 0x2421e5d236704588UL, 0x887af7d4dd482328UL,
    0xa5e9c291f6119b27UL, 0xbdc52b2676a4b4aaUL, 0x64832009d29bcf57UL, 0x09c4155174a552ccUL,
    0x463f9ee03d290810UL, 0xc810936e64982542UL, 0x043b1c289f7bc3acUL,
    // Round 1
    0x673655aae8be5a8bUL, 0xd510fe714f39fa10UL, 0x2c68a099b51c9e73UL, 0xa667bfa9aa96999dUL,
    0x4d67e72f063e2108UL, 0xf84dde3e6acda179UL, 0x40f9cc8c08f80981UL, 0x5ead032050097142UL,
    0x6591b02092d671bbUL, 0x00e18c71963dd1b7UL, 0x8a21bcd24a14218aUL,
    // Round 2
    0x202800f4addbdc87UL, 0xe4b5bdb1cc3504ffUL, 0xbe32b32a825596e7UL, 0x8e0f68c5dc223b9aUL,
    0x58022d9e1c256ce3UL, 0x584d29227aa073acUL, 0x8b9352ad04bef9e7UL, 0xaead42a3f445ecbfUL,
    0x3c667a1d833a3ccaUL, 0xda6f61838efa1ffeUL, 0xe8f749470bd7c446UL,
    // Round 3
    0xc5b85bab9e5b3869UL, 0x45245258aec51cf7UL, 0x16e6b8e68b931830UL, 0xe2ae0f051418112cUL,
    0x0470e26a0093a65bUL, 0x6bef71973a8146edUL, 0x119265be51812dafUL, 0xb0be7356254bea2eUL,
    0x8584defff7589bd7UL, 0x3c5fe4aeb1fb52baUL, 0x9e7cd88acf543a5eUL,
    // Round 4
    0x179be4bba87f0a8cUL, 0xacf63d95d8887355UL, 0x6696670196b0074fUL, 0xd99ddf1fe75085f9UL,
    0xc2597881fef0283bUL, 0xcf48395ee6c54f14UL, 0x15226a8e4cd8d3b6UL, 0xc053297389af5d3bUL,
    0x2c08893f0d1580e2UL, 0x0ed3cbcff6fcc5baUL, 0xc82f510ecf81f6d0UL,
    // Round 5
    0x94b06183acb715ccUL, 0x500392ed0d431137UL, 0x861cc95ad5c86323UL, 0x05830a443f86c4acUL,
    0x3b68225874a20a7cUL, 0x10b3309838e236fbUL, 0x9b77fc8bcd559e2cUL, 0xbdecf5e0cb9cb213UL,
    0x30276f1221ace5faUL, 0x7935dd342764a144UL, 0xeac6db520bb03708UL,
    // Round 6
    0x7186a80551025f8fUL, 0x622247557e9b5371UL, 0xc4cbe326d1ad9742UL, 0x55f1523ac6a23ea2UL,
    0xa13dfe77a3d52f53UL, 0xe30750b6301c0452UL, 0x08bd488070a3a32bUL, 0xcd800caef5b72ae3UL,
    0x83329c90f04233ceUL, 0xb5b99e6664a0a3eeUL, 0x6b0731849e200a7fUL,
    // Round 7
    0xec3fabc192b01799UL, 0x382b38cee8ee5375UL, 0x3bfb6c3f0e616572UL, 0x514abd0cf6c7bc86UL,
    0x47521b1361dcc546UL, 0x178093843f863d14UL, 0xad1003c5d28918e7UL, 0x738450e42495bc81UL,
    0xaf947c59af5e4047UL, 0x4653fb0685084ef2UL, 0x057fde2062ae35bfUL,
    // Round 8
    0xe376678d843ce55eUL, 0x66f3860d7514e7fcUL, 0x7817f3dfff8b4ffaUL, 0x3929624a9def725bUL,
    0x0126ca37f215a80aUL, 0xfce2f5d02762a303UL, 0x1bc927375febbad7UL, 0x85b481e5243f60bfUL,
    0x2d3c5f42a39c91a0UL, 0x0811719919351ae8UL, 0xf669de0add993131UL,
    // Round 9
    0x7de38bae084da92dUL, 0x5b848442237e8a9bUL, 0xf6c705da84d57310UL, 0x31e6a4bdb6a49017UL,
    0x889489706e5c5c0fUL, 0x0e4a205459692a1bUL, 0xbac3fa75ee26f299UL, 0x5f5894f4057d755eUL,
    0xb0dc3ecd724bb076UL, 0x5e34d8554a6452baUL, 0x04f78fd8c1fdcc5fUL,
    // Round 10
    0x4dd19c38779512eaUL, 0xdb79ba02704620e9UL, 0x92a29a3675a5d2beUL, 0xd5177029fe495166UL,
    0xd32b3298a13330c1UL, 0x251c4a3eb2c5f8fdUL, 0xe1c48b26e0d98825UL, 0x3301d3362a4ffccbUL,
    0x09bb6c88de8cd178UL, 0xdc05b676564f538aUL, 0x60192d883e473feeUL,
    // Round 11
    0x16b9774801ac44a0UL, 0x3cb8411e786d3c8eUL, 0xa86e9cf505072491UL, 0x0178928152e109aeUL,
    0x5317b905a6e1ab7bUL, 0xda20b3be7f53d59fUL, 0xcb97dedecebee9adUL, 0x4bd545218c59f58dUL,
    0x77dc8d856c05a44aUL, 0x87948589e4f243fdUL, 0x7e5217af969952c2UL,
    // Round 12
    0xbc58987d06a84e4dUL, 0x0b5d420244c9cae3UL, 0xa3c4711b938c02c0UL, 0x3aace640a3e03990UL,
    0x865a0f3249aacd8aUL, 0x8d00b2a7dbed06c7UL, 0x6eacb905beb7e2f8UL, 0x045322b216ec3ec7UL,
    0xeb9de00d594828e6UL, 0x088c5f20df9e5c26UL, 0xf555f4112b19781fUL,
    // Round 13
    0xa8cedbff1813d3a7UL, 0x50dcaee0fd27d164UL, 0xf1cb02417e23bd82UL, 0xfaf322786e2abe8bUL,
    0x937a4315beb5d9b6UL, 0x1b18992921a11d85UL, 0x7d66c4368b3c497bUL, 0x0e7946317a6b4e99UL,
    0xbe4430134182978bUL, 0x3771e82493ab262dUL, 0xa671690d8095ce82UL,
    // Round 14
    0xb035585f6e929d9dUL, 0xba1579c7e219b954UL, 0xcb201cf846db4ba3UL, 0x287bf9177372cf45UL,
    0xa350e4f61147d0a6UL, 0xd5d0ecfb50bcff99UL, 0x2e166aa6c776ed21UL, 0xe1e66c991990e282UL,
    0x662b329b01e7bb38UL, 0x8aa674b36144d9a9UL, 0xcbabf78f97f95e65UL,
    // Round 15
    0xeec24b15a06b53feUL, 0xc8a7aa07c5633533UL, 0xefe9c6fa4311ad51UL, 0xb9173f13977109a1UL,
    0x69ce43c9cc94aedcUL, 0xecf623c9cd118815UL, 0x28625def198c33c7UL, 0xccfc5f7de5c3636aUL,
    0xf5e6c40f1621c299UL, 0xcec0e58c34cb64b1UL, 0xa868ea113387939fUL,
    // Round 16
    0xd8dddbdc5ce4ef45UL, 0xacfc51de8131458cUL, 0x146bb3c0fe499ac0UL, 0x9e65309f15943903UL,
    0x80d0ad980773aa70UL, 0xf97817d4ddbf0607UL, 0xe4626620a75ba276UL, 0x0dfdc7fd6fc74f66UL,
    0xf464864ad6f2bb93UL, 0x02d55e52a5d44414UL, 0xdd8de62487c40925UL,
    // Round 17
    0xc15acf44759545a3UL, 0xcbfdcf39869719d4UL, 0x33f62042e2f80225UL, 0x2599c5ead81d8fa3UL,
    0x0b306cb6c1d7c8d0UL, 0x658c80d3df3729b1UL, 0xe8d1b2b21b41429cUL, 0xa1b67f09d4b3ccb8UL,
    0x0e1adf8b84437180UL, 0x0d593a5e584af47bUL, 0xa023d94c56e151c7UL,
    // Round 18
    0x49026cc3a4afc5a6UL, 0xe06dff00ab25b91bUL, 0x0ab38c561e8850ffUL, 0x92c3c8275e105eebUL,
    0xb65256e546889bd0UL, 0x3c0468236ea142f6UL, 0xee61766b889e18f2UL, 0xa206f41b12c30415UL,
    0x02fe9d756c9f12d1UL, 0xe9633210630cbf12UL, 0x1ffea9fe85a0b0b1UL,
    // Round 19
    0x81d1ae8cc50240f3UL, 0xf4c77a079a4607d7UL, 0xed446b2315e3efc1UL, 0x0b0a6b70915178c3UL,
    0xb11ff3e089f15d9aUL, 0x1d4dba0b7ae9cc18UL, 0x65d74e2f43b48d05UL, 0xa2df8c6b8ae0804aUL,
    0xa4e6f0a8c33348a6UL, 0xc0a26efc7be5669bUL, 0xa6b6582c547d0d60UL,
    // Round 20
    0x84afc741f1c13213UL, 0x2f8f43734fc906f3UL, 0xde682d72da0a02d9UL, 0x0bb005236adb9ef2UL,
    0x5bdf35c10a8b5624UL, 0x0739a8a343950010UL, 0x52f515f44785cfbcUL, 0xcbaf4e5d82856c60UL,
    0xac9ea09074e3e150UL, 0x8f0fa011a2035fb0UL, 0x1a37905d8450904aUL,
    // Round 21
    0x3abeb80def61cc85UL, 0x9d19c9dd4eac4133UL, 0x075a652d9641a985UL, 0x9daf69ae1b67e667UL,
    0x364f71da77920a18UL, 0x50bd769f745c95b1UL, 0xf223d1180dbbf3fcUL, 0x2f885e584e04aa99UL,
    0xb69a0fa70aea684aUL, 0x09584acaa6e062a0UL, 0x0bc051640145b19bUL,
};

// V vectors for updating state[1..11] in each partial round (22 × 11, flattened)
constant uint64_t FP_VS[242] = {
    // Round 0
    0x94877900674181c3UL, 0xc6c67cc37a2a2bbdUL, 0xd667c2055387940fUL, 0x0ba63a63e94b5ff0UL,
    0x99460cc41b8f079fUL, 0x7ff02375ed524bb3UL, 0xea0870b47a8caf0eUL, 0xabcad82633b7bc9dUL,
    0x3b8d135261052241UL, 0xfb4515f5e5b0d539UL, 0x3ee8011c2b37f77cUL,
    // Round 1
    0x0adef3740e71c726UL, 0xa37bf67c6f986559UL, 0xc6b16f7ed4fa1b00UL, 0x6a065da88d8bfc3cUL,
    0x4cabc0916844b46fUL, 0x407faac0f02e78d1UL, 0x07a786d9cf0852cfUL, 0x42433fb6949a629aUL,
    0x891682a147ce43b0UL, 0x26cfd58e7b003b55UL, 0x2bbf0ed7b657acb3UL,
    // Round 2
    0x481ac7746b159c67UL, 0xe367de32f108e278UL, 0x73f260087ad28becUL, 0x5cfc82216bc1bdcaUL,
    0xcaccc870a2663a0eUL, 0xdb69cd7b4298c45dUL, 0x7bc9e0c57243e62dUL, 0x3cc51c5d368693aeUL,
    0x366b4e8cc068895bUL, 0x2bd18715cdabbca4UL, 0xa752061c4f33b8cfUL,
    // Round 3
    0xb22d2432b72d5098UL, 0x9e18a487f44d2fe4UL, 0x4b39e14ce22abd3cUL, 0x9e77fde2eb315e0dUL,
    0xca5e0385fe67014dUL, 0x0c2cb99bf1b6bddbUL, 0x99ec1cd2a4460bfeUL, 0x8577a815a2ff843fUL,
    0x7d80a6b4fd6518a5UL, 0xeb6c67123eab62cbUL, 0x8f7851650eca21a5UL,
    // Round 4
    0x11ba9a1b81718c2aUL, 0x9f7d798a3323410cUL, 0xa821855c8c1cf5e5UL, 0x535e8d6fac0031b2UL,
    0x404e7c751b634320UL, 0xa729353f6e55d354UL, 0x4db97d92e58bb831UL, 0xb53926c27897bf7dUL,
    0x965040d52fe115c5UL, 0x9565fa41ebd31fd7UL, 0xaae4438c877ea8f4UL,
    // Round 5
    0x37f4e36af6073c6eUL, 0x4edc0918210800e9UL, 0xc44998e99eae4188UL, 0x9f4310d05d068338UL,
    0x9ec7fe4350680f29UL, 0xc5b2c1fdc0b50874UL, 0xa01920c5ef8b2ebeUL, 0x59fa6f8bd91d58baUL,
    0x8bfc9eb89b515a82UL, 0xbe86a7a2555ae775UL, 0xcbb8bbaa3810babfUL,
    // Round 6
    0x577f9a9e7ee3f9c2UL, 0x88c522b949ace7b1UL, 0x82f07007c8b72106UL, 0x8283d37c6675b50eUL,
    0x98b074d9bbac1123UL, 0x75c56fb7758317c1UL, 0xfed24e206052bc72UL, 0x26d7c3d1bc07dae5UL,
    0xf88c5e441e28dbb4UL, 0x4fe27f9f96615270UL, 0x514d4ba49c2b14feUL,
    // Round 7
    0xf02a3ac068ee110bUL, 0x0a3630dafb8ae2d7UL, 0xce0dc874eaf9b55cUL, 0x9a95f6cff5b55c7eUL,
    0x626d76abfed00c7bUL, 0xa0c1cf1251c204adUL, 0xdaebd3006321052cUL, 0x3d4bd48b625a8065UL,
    0x7f1e584e071f6ed2UL, 0x720574f0501caed3UL, 0xe3260ba93d23540aUL,
    // Round 8
    0xab1cbd41d8c1e335UL, 0x9322ed4c0bc2df01UL, 0x51c3c0983d4284e5UL, 0x94178e291145c231UL,
    0xfd0f1a973d6b2085UL, 0xd427ad96e2b39719UL, 0x8a52437fecaac06bUL, 0xdc20ee4b8c4c9a80UL,
    0xa2c98e9549da2100UL, 0x1603fe12613db5b6UL, 0x0e174929433c5505UL,
    // Round 9
    0x3d4eab2b8ef5f796UL, 0xcfff421583896e22UL, 0x4143cb32d39ac3d9UL, 0x22365051b78a5b65UL,
    0x6f7fd010d027c9b6UL, 0xd9dd36fba77522abUL, 0xa44cf1cb33e37165UL, 0x3fc83d3038c86417UL,
    0xc4588d418e88d270UL, 0xce1320f10ab80fe2UL, 0xdb5eadbbec18de5dUL,
    // Round 10
    0x1183dfce7c454afdUL, 0x21cea4aa3d3ed949UL, 0x0fce6f70303f2304UL, 0x19557d34b55551beUL,
    0x4c56f689afc5bbc9UL, 0xa1e920844334f944UL, 0xbad66d423d2ec861UL, 0xf318c785dc9e0479UL,
    0x99e2032e765ddd81UL, 0x400ccc9906d66f45UL, 0xe1197454db2e0dd9UL,
    // Round 11
    0x84d1ecc4d53d2ff1UL, 0xd8af8b9ceb4e11b6UL, 0x335856bb527b52f4UL, 0xc756f17fb59be595UL,
    0xc0654e4ea5553a78UL, 0x9e9a46b61f2ea942UL, 0x14fc8b5b3b809127UL, 0xd7009f0f103be413UL,
    0x3e0ee7b7a9fb4601UL, 0xa74e888922085ed7UL, 0xe80a7cde3d4ac526UL,
    // Round 12
    0x238aa6daa612186dUL, 0x9137a5c630bad4b4UL, 0xc7db3817870c5edaUL, 0x217e4f04e5718dc9UL,
    0xcae814e2817bd99dUL, 0xe3292e7ab770a8baUL, 0x7bb36ef70b6b9482UL, 0x3c7835fb85bca2d3UL,
    0xfe2cdf8ee3c25e86UL, 0x61b3915ad7274b20UL, 0xeab75ca7c918e4efUL,
    // Round 13
    0xd6e15ffc055e154eUL, 0xec67881f381a32bfUL, 0xfbb1196092bf409cUL, 0xdc9d2e07830ba226UL,
    0x0698ef3245ff7988UL, 0x194fae2974f8b576UL, 0x7a5d9bea6ca4910eUL, 0x7aebfea95ccdd1c9UL,
    0xf9bd38a67d5f0e86UL, 0xfa65539de65492d8UL, 0xf0dfcbe7653ff787UL,
    // Round 14
    0x0bd87ad390420258UL, 0x0ad8617bca9e33c8UL, 0x0c00ad377a1e2666UL, 0x0ac6fc58b3f0518fUL,
    0x0c0cc8a892cc4173UL, 0x0c210accb117bc21UL, 0x0b73630dbb46ca18UL, 0x0c8be4920cbd4a54UL,
    0x0bfe877a21be1690UL, 0x0ae790559b0ded81UL, 0x0bf50db2f8d6ce31UL,
    // Round 15
    0x000cf29427ff7c58UL, 0x000bd9b3cf49eec8UL, 0x000d1dc8aa81fb26UL, 0x000bc792d5c394efUL,
    0x000d2ae0b2266453UL, 0x000d413f12c496c1UL, 0x000c84128cfed618UL, 0x000db5ebd48fc0d4UL,
    0x000d1b77326dcb90UL, 0x000beb0ccc145421UL, 0x000d10e5b22b11d1UL,
    // Round 16
    0x00000e24c99adad8UL, 0x00000cf389ed4bc8UL, 0x00000e580cbf6966UL, 0x00000cde5fd7e04fUL,
    0x00000e63628041b3UL, 0x00000e7e81a87361UL, 0x00000dabe78f6d98UL, 0x00000efb14cac554UL,
    0x00000e5574743b10UL, 0x00000d05709f42c1UL, 0x00000e4690c96af1UL,
    // Round 17
    0x0000000f7157bc98UL, 0x0000000e3006d948UL, 0x0000000fa65811e6UL, 0x0000000e0d127e2fUL,
    0x0000000fc18bfe53UL, 0x0000000fd002d901UL, 0x0000000eed6461d8UL, 0x0000001068562754UL,
    0x0000000fa0236f50UL, 0x0000000e3af13ee1UL, 0x0000000fa460f6d1UL,
    // Round 18
    0x0000000011131738UL, 0x000000000f56d588UL, 0x0000000011050f86UL, 0x000000000f848f4fUL,
    0x00000000111527d3UL, 0x00000000114369a1UL, 0x00000000106f2f38UL, 0x0000000011e2ca94UL,
    0x00000000110a29f0UL, 0x000000000fa9f5c1UL, 0x0000000010f625d1UL,
    // Round 19
    0x000000000011f718UL, 0x000000000010b6c8UL, 0x0000000000134a96UL, 0x000000000010cf7fUL,
    0x0000000000124d03UL, 0x000000000013f8a1UL, 0x0000000000117c58UL, 0x0000000000132c94UL,
    0x0000000000134fc0UL, 0x000000000010a091UL, 0x0000000000128961UL,
    // Round 20
    0x0000000000001300UL, 0x0000000000001750UL, 0x000000000000114eUL, 0x000000000000131fUL,
    0x000000000000167bUL, 0x0000000000001371UL, 0x0000000000001230UL, 0x000000000000182cUL,
    0x0000000000001368UL, 0x0000000000000f31UL, 0x00000000000015c9UL,
    // Round 21
    0x0000000000000014UL, 0x0000000000000022UL, 0x0000000000000012UL, 0x0000000000000027UL,
    0x000000000000000dUL, 0x000000000000000dUL, 0x000000000000001cUL, 0x0000000000000002UL,
    0x0000000000000010UL, 0x0000000000000029UL, 0x000000000000000fUL,
};

// Initial matrix for state[1..11] transformation (11×11, row-major)
constant uint64_t FP_INIT_MAT[121] = {
    // Row 0
    0x80772dc2645b280bUL, 0xdc927721da922cf8UL, 0xc1978156516879adUL, 0x90e80c591f48b603UL,
    0x3a2432625475e3aeUL, 0x00a2d4321cca94feUL, 0x77736f524010c932UL, 0x904d3f2804a36c54UL,
    0xbf9b39e28a16f354UL, 0x3a1ded54a6cd058bUL, 0x42392870da5737cfUL,
    // Row 1
    0xe796d293a47a64cbUL, 0xb124c33152a2421aUL, 0x0ee5dc0ce131268aUL, 0xa9032a52f930fae6UL,
    0x7e33ca8c814280deUL, 0xad11180f69a8c29eUL, 0xc75ac6d5b5a10ff3UL, 0xf0674a8dc5a387ecUL,
    0xb36d43120eaa5e2bUL, 0x6f232aab4b533a25UL, 0x3a1ded54a6cd058bUL,
    // Row 2
    0xdcedab70f40718baUL, 0x14a4a64da0b2668fUL, 0x4715b8e5ab34653bUL, 0x1e8916a99c93a88eUL,
    0xbba4b5d86b9a3b2cUL, 0xe76649f9bd5d5c2eUL, 0xaf8e2518a1ece54dUL, 0xdcda1344cdca873fUL,
    0xcd080204256088e5UL, 0xb36d43120eaa5e2bUL, 0xbf9b39e28a16f354UL,
    // Row 3
    0xf4a437f2888ae909UL, 0xc537d44dc2875403UL, 0x7f68007619fd8ba9UL, 0xa4911db6a32612daUL,
    0x2f7e9aade3fdaec1UL, 0xe7ffd578da4ea43dUL, 0x43a608e7afa6b5c2UL, 0xca46546aa99e1575UL,
    0xdcda1344cdca873fUL, 0xf0674a8dc5a387ecUL, 0x904d3f2804a36c54UL,
    // Row 4
    0xf97abba0dffb6c50UL, 0x5e40f0c9bb82aab5UL, 0x5996a80497e24a6bUL, 0x07084430a7307c9aUL,
    0xad2f570a5b8545aaUL, 0xab7f81fef4274770UL, 0xcb81f535cf98c9e9UL, 0x43a608e7afa6b5c2UL,
    0xaf8e2518a1ece54dUL, 0xc75ac6d5b5a10ff3UL, 0x77736f524010c932UL,
    // Row 5
    0x7f8e41e0b0a6cdffUL, 0x4b1ba8d40afca97dUL, 0x623708f28fca70e8UL, 0xbf150dc4914d380fUL,
    0xc26a083554767106UL, 0x753b8b1126665c22UL, 0xab7f81fef4274770UL, 0xe7ffd578da4ea43dUL,
    0xe76649f9bd5d5c2eUL, 0xad11180f69a8c29eUL, 0x00a2d4321cca94feUL,
    // Row 6
    0x726af914971c1374UL, 0x1d7f8a2cce1a9d00UL, 0x18737784700c75cdUL, 0x7fb45d605dd82838UL,
    0x862361aeab0f9b6eUL, 0xc26a083554767106UL, 0xad2f570a5b8545aaUL, 0x2f7e9aade3fdaec1UL,
    0xbba4b5d86b9a3b2cUL, 0x7e33ca8c814280deUL, 0x3a2432625475e3aeUL,
    // Row 7
    0x64dd936da878404dUL, 0x4db9a2ead2bd7262UL, 0xbe2e19f6d07f1a83UL, 0x02290fe23c20351aUL,
    0x7fb45d605dd82838UL, 0xbf150dc4914d380fUL, 0x07084430a7307c9aUL, 0xa4911db6a32612daUL,
    0x1e8916a99c93a88eUL, 0xa9032a52f930fae6UL, 0x90e80c591f48b603UL,
    // Row 8
    0x85418a9fef8a9890UL, 0xd8a2eb7ef5e707adUL, 0xbfe85ababed2d882UL, 0xbe2e19f6d07f1a83UL,
    0x18737784700c75cdUL, 0x623708f28fca70e8UL, 0x5996a80497e24a6bUL, 0x7f68007619fd8ba9UL,
    0x4715b8e5ab34653bUL, 0x0ee5dc0ce131268aUL, 0xc1978156516879adUL,
    // Row 9
    0x156048ee7a738154UL, 0x91f7562377e81df5UL, 0xd8a2eb7ef5e707adUL, 0x4db9a2ead2bd7262UL,
    0x1d7f8a2cce1a9d00UL, 0x4b1ba8d40afca97dUL, 0x5e40f0c9bb82aab5UL, 0xc537d44dc2875403UL,
    0x14a4a64da0b2668fUL, 0xb124c33152a2421aUL, 0xdc927721da922cf8UL,
    // Row 10
    0xd841e8ef9dde8ba0UL, 0x156048ee7a738154UL, 0x85418a9fef8a9890UL, 0x64dd936da878404dUL,
    0x726af914971c1374UL, 0x7f8e41e0b0a6cdffUL, 0xf97abba0dffb6c50UL, 0xf4a437f2888ae909UL,
    0xdcedab70f40718baUL, 0xe796d293a47a64cbUL, 0x80772dc2645b280bUL,
};

// =============================================================================
// Poseidon permutation — operates on a 12-element Goldilocks state
// =============================================================================

// S-box: x^7 = x * x^2 * x^4  (3 multiplications)
inline uint64_t poseidon_sbox(uint64_t x) {
    uint64_t x2 = gl_mul(x, x);
    uint64_t x4 = gl_mul(x2, x2);
    uint64_t x3 = gl_mul(x, x2);
    return gl_mul(x3, x4);
}

// MDS row product using u128 accumulation (modulo-free).
// Computes row r of: M*state, where M = circ(MDS_CIRC) + diag(MDS_DIAG).
inline uint64_t mds_row(uint r, thread uint64_t* state) {
    uint64_t lo = 0;
    uint64_t hi = 0;

    for (uint i = 0; i < SPONGE_WIDTH; i++) {
        // Replace expensive % with conditional subtract
        uint idx = i + r;
        if (idx >= SPONGE_WIDTH) idx -= SPONGE_WIDTH;
        acc_mul(lo, hi, state[idx], MDS_CIRC[i]);
    }

    // Diagonal term (only MDS_DIAG[0] = 8 is nonzero)
    if (r == 0) {
        acc_mul(lo, hi, state[0], 8);
    }

    return gl_reduce_u128(lo, hi);
}

// Apply MDS layer to state in-place
inline void mds_layer(thread uint64_t* state) {
    uint64_t out[12];
    for (uint r = 0; r < SPONGE_WIDTH; r++) {
        out[r] = mds_row(r, state);
    }
    for (uint i = 0; i < SPONGE_WIDTH; i++) {
        state[i] = out[i];
    }
}

// Add round constants to state (for full rounds only)
inline void constant_layer(thread uint64_t* state, uint round) {
    uint base = round * SPONGE_WIDTH;
    for (uint i = 0; i < SPONGE_WIDTH; i++) {
        state[i] = gl_add(state[i], RC[base + i]);
    }
}

// Full round: add constants, S-box all elements, MDS mix
inline void full_round(thread uint64_t* state, uint round) {
    constant_layer(state, round);
    for (uint i = 0; i < SPONGE_WIDTH; i++) {
        state[i] = poseidon_sbox(state[i]);
    }
    mds_layer(state);
}

// =============================================================================
// Poseidon permutation — fast partial rounds (Plonky2 technique)
//
// 4 full rounds + 22 fast partial rounds + 4 full rounds = 30 total.
// Fast partial rounds use sparse MDS factorization to reduce per-round MDS
// from O(n^2) to O(n) multiplications.
// =============================================================================

inline void poseidon_permute(thread uint64_t* state) {
    // === Phase 1: First 4 full rounds (standard) ===
    for (uint round = 0; round < HALF_N_FULL_ROUNDS; round++) {
        full_round(state, round);
    }

    // === Phase 2: Fast partial rounds ===

    // 2a. Add first partial round transformed constants
    for (uint i = 0; i < SPONGE_WIDTH; i++) {
        state[i] = gl_add(state[i], FP_FIRST_RC[i]);
    }

    // 2b. Apply initial 11×11 matrix to state[1..11] (transposed access)
    // Plonky2 computes: result[c+1] = sum_r state[r+1] * INIT_MAT[r][c]
    // state[0] passes through unchanged.
    {
        uint64_t tmp[12];
        tmp[0] = state[0];
        for (uint c = 0; c < 11; c++) {
            uint64_t sum = 0;
            for (uint r = 0; r < 11; r++) {
                sum = gl_add(sum, gl_mul(state[r + 1], FP_INIT_MAT[r * 11 + c]));
            }
            tmp[c + 1] = sum;
        }
        for (uint i = 0; i < SPONGE_WIDTH; i++) {
            state[i] = tmp[i];
        }
    }

    // 2c. 22 partial rounds with sparse MDS
    for (uint i = 0; i < N_PARTIAL_ROUNDS; i++) {
        // S-box on state[0] only
        state[0] = poseidon_sbox(state[0]);

        // Add per-round scalar constant to state[0]
        state[0] = gl_add(state[0], FP_RC[i]);

        // Save pre-update state[0]
        uint64_t s0 = state[0];

        // Compute new state[0] = mds0to0 * s0 + dot(w_hat, state[1..11])
        // where mds0to0 = MDS_CIRC[0] + MDS_DIAG[0] = 17 + 8 = 25
        uint64_t d = gl_mul(s0, 25);
        uint base = i * 11;
        for (uint j = 0; j < 11; j++) {
            d = gl_add(d, gl_mul(state[j + 1], FP_W_HATS[base + j]));
        }

        // Update state[1..11]: state[j+1] += s0 * VS[j]
        for (uint j = 0; j < 11; j++) {
            state[j + 1] = gl_add(state[j + 1], gl_mul(s0, FP_VS[base + j]));
        }

        state[0] = d;
    }

    // === Phase 3: Last 4 full rounds (standard) ===
    for (uint round = HALF_N_FULL_ROUNDS + N_PARTIAL_ROUNDS; round < N_ROUNDS; round++) {
        full_round(state, round);
    }
}

// =============================================================================
// Kernel 1: Hash leaves (sponge absorption without padding)
//
// Each thread hashes one row of u64 field elements into a [u8; 32] node.
// The node stores the Poseidon hash (one u64) in the first 8 bytes (LE),
// with the remaining 24 bytes zeroed.
//
// Input: flat row-major u64 array, row i starts at lde_data[i * num_cols]
// Output: 32 bytes per leaf at output[gid * 32]
//
// Sponge: rate=8, capacity=4. Absorb 8 elements at a time, permute.
// No padding (input length is known and fixed).
// =============================================================================

[[kernel]] void poseidon_hash_leaves(
    device const ulong* lde_data    [[ buffer(0) ]],
    device uchar*        output     [[ buffer(1) ]],
    constant uint&       num_cols   [[ buffer(2) ]],
    constant uint&       num_rows   [[ buffer(3) ]],
    uint gid                        [[ thread_position_in_grid ]]
) {
    if (gid >= num_rows) return;

    uint64_t state[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    uint data_offset = gid * num_cols;

    // Absorb in blocks of SPONGE_RATE (8) elements
    uint col = 0;
    while (col + SPONGE_RATE <= num_cols) {
        for (uint i = 0; i < SPONGE_RATE; i++) {
            state[i] = gl_add(state[i], lde_data[data_offset + col + i]);
        }
        poseidon_permute(state);
        col += SPONGE_RATE;
    }

    // Remaining elements (partial block)
    uint remaining = num_cols - col;
    if (remaining > 0) {
        for (uint i = 0; i < remaining; i++) {
            state[i] = gl_add(state[i], lde_data[data_offset + col + i]);
        }
        poseidon_permute(state);
    }

    // Output: write state[0] as 8 bytes LE + 24 zero bytes = 32-byte node
    uint64_t hash_val = gl_canon(state[0]);
    device ulong* out64 = (device ulong*)(output + gid * 32);
    out64[0] = hash_val;
    out64[1] = 0;
    out64[2] = 0;
    out64[3] = 0;
}

// =============================================================================
// Kernel 2: Hash pairs of child nodes into parent nodes
//
// Each thread takes two consecutive 32-byte child nodes and produces one
// 32-byte parent: parent[gid] = Poseidon_hash(child[2*gid], child[2*gid+1])
//
// Child nodes store a Goldilocks element in the first 8 bytes (LE).
// Hash(a, b): state = [a, b, 0, ..., 0], permute, output state[0].
// =============================================================================

[[kernel]] void poseidon_hash_pairs(
    device const uchar* children   [[ buffer(0) ]],
    device uchar*       parents    [[ buffer(1) ]],
    constant uint&      num_pairs  [[ buffer(2) ]],
    uint gid                       [[ thread_position_in_grid ]]
) {
    if (gid >= num_pairs) return;

    // Read two child nodes — each is 32 bytes, hash value in first 8 bytes (LE u64)
    device const ulong* left64  = (device const ulong*)(children + gid * 64);
    device const ulong* right64 = (device const ulong*)(children + gid * 64 + 32);

    uint64_t left_val  = left64[0];
    uint64_t right_val = right64[0];

    // Poseidon hash(a, b): place in state[0..1], zero rest, permute
    uint64_t state[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    state[0] = left_val;
    state[1] = right_val;
    poseidon_permute(state);

    // Write parent node: state[0] in first 8 bytes (LE), zero rest
    uint64_t hash_val = gl_canon(state[0]);
    device ulong* out64 = (device ulong*)(parents + gid * 32);
    out64[0] = hash_val;
    out64[1] = 0;
    out64[2] = 0;
    out64[3] = 0;
}
