<div align="center">

# Lambdaworks Merklee Tree CLI

Simple Merkle Tree CLI that uses [Poseidon hash](https://www.poseidon-hash.info/).

</div>

### Usage:

### To create a Merkle Tree and save it to a file you can use:

```bash
cargo run --release generate-tree <TREE_PATH>
```

The format of a tree `csv` file is as follows:
```
elem1;elem2;...;elemN
```
For example, the provided **`sample_tree.csv`** looks like this:
```
0x12345;0x6789A;0xBCDEF
```

**`generate-tree` example:**

```bash
cargo run --release generate-tree sample_tree.csv
```

### To generate proof for a Merkle Tree you can use: 

```bash
cargo run --release generate-proof <TREE_PATH> <POSITION>
```

**`generate-proof` example:**

```bash
cargo run --release generate-proof sample_tree.csv 0
```

### To verify a proof you can use:

```bash
cargo run --release verify-proof <ROOT_PATH> <INDEX> <PROOF_PATH> <LEAF_PATH>
```

The format of a root `txt` file is a simple text file which only containts the root as a hex string. Using the root that yields the merkle tree generated from the `sample_tree` provided, **`root.txt`** would look like this:
```
0xa3bbbb9eac9f79d18862b802ea79f87e75efc37d4f4af4464976784c14a851b69c09aa04b1e8a8d1eb9825b713dc6ca
```

Likewise, the format of a leaf `txt` file is a simple text file which only contains the leaf as a hex string. Using the first element (index 0) of the provided `sample_tree.csv` as out leaf, **`leaf.txt`** would look like this:
```
0x12345
```

**`verify-proof` example:**

```bash
cargo run --release verify-proof root.txt 0 sample_tree_proof_0.json leaf.txt
```

## 📚 References

The following links, repos and projects have been important in the development of this library and we want to thank and acknowledge them. 

- [Clap4](https://epage.github.io/blog/2022/09/clap4/)
- [Starkware](https://starkware.co/)
- [Winterfell](https://github.com/facebook/winterfell)
- [Anatomy of a Stark](https://aszepieniec.github.io/stark-anatomy/overview)
- [Giza](https://github.com/maxgillett/giza)
- [Ministark](https://github.com/andrewmilson/ministark)
- [Sandstorm](https://github.com/andrewmilson/sandstorm)
- [STARK-101](https://starkware.co/stark-101/)
- [Risc0](https://github.com/risc0/risc0)
- [Neptune](https://github.com/Neptune-Crypto)
- [Summary on FRI low degree test](https://eprint.iacr.org/2022/1216)
- [STARKs paper](https://eprint.iacr.org/2018/046)
- [DEEP FRI](https://eprint.iacr.org/2019/336)
- [BrainSTARK](https://aszepieniec.github.io/stark-brainfuck/)
- [Plonky2](https://github.com/mir-protocol/plonky2)
- [Aztec](https://github.com/AztecProtocol)
- [Arkworks](https://github.com/arkworks-rs)
- [Thank goodness it's FRIday](https://vitalik.ca/general/2017/11/22/starks_part_2.html)
- [Diving DEEP FRI](https://blog.lambdaclass.com/diving-deep-fri/)
- [Periodic constraints](https://blog.lambdaclass.com/periodic-constraints-and-recursion-in-zk-starks/)
- [Chiplets Miden VM](https://wiki.polygon.technology/docs/miden/design/chiplets/main/)
- [Valida](https://github.com/valida-xyz/valida/tree/main)
- [Solidity Verifier](https://github.com/starkware-libs/starkex-contracts/tree/master/evm-verifier/solidity/contracts/cpu)
- [CAIRO verifier](https://github.com/starkware-libs/cairo-lang/tree/master/src/starkware/cairo/stark_verifier)
- [EthSTARK](https://github.com/starkware-libs/ethSTARK/tree/master)
- [CAIRO whitepaper](https://eprint.iacr.org/2021/1063.pdf)
- [Gnark](https://github.com/Consensys/gnark)
