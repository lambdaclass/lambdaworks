use ark_bn254::Fr as ArkBn254Fr;

fn main() {
    let a = ArkBn254Fr::from(0xDEADBEEFu64);
    let b = ArkBn254Fr::from(0xCAFEBABEu64);

    for _ in 0..100_000 {
        std::hint::black_box(std::hint::black_box(a) * std::hint::black_box(b));
    }
}
