# Hash functions

This folder contains hash functions that are typically used in non-interactive proof systems. The hash functions we have implemented are:
- [Monolith](./monolith/mod.rs)
- [Poseidon](./poseidon/)
- [Pedersen](./pedersen/)
- [Rescue](./rescue/)

Rescue contains two hash functions:
- **RPO** (Rescue Prime Optimized): https://eprint.iacr.org/2022/1577
- **RPX** (Rescue Prime eXtension / XHash-12): https://eprint.iacr.org/2023/1045

RPX is ~2x faster than RPO by using cubic extension field arithmetic in the extension rounds.

Pedersen is based on elliptic curves, while [Monolith](https://eprint.iacr.org/2023/1025), [Poseidon](https://eprint.iacr.org/2019/458.pdf) and [Rescue Prime](https://eprint.iacr.org/2020/1143) are algebraic hash functions.

For an introduction to hash functions, see [this intro](https://blog.alignedlayer.com/introduction-hash-functions-in-cryptography/) and [its follow-up](https://blog.alignedlayer.com/design-strategies-how-to-construct-a-hashing-mode-2/).