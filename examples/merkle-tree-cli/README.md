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
