<div align="center">

# Lambdaworks Merklee Tree CLI

</div>

### Usage:

**To create a Merkle Tree and save it to a file you can use:**

```bash
cargo run generate-merkle-tree <TREE_PATH>
```

For example:

```bash
cargo run generate-merkle-tree sample_tree.csv
```


**To generate proof for a Merkle Tree you can use:**

```bash
cargo run generate-merkle-tree <TREE_PATH> <POSITION>
```

For example:

```bash
cargo run generate-merkle-tree sample_tree.csv 0
```

**To verify a proof you can use:**

```bash
cargo run generate-merkle-tree <ROOT_PATH> <INDEX> <PROOF_PATH> <LEAF_PATH>
```

For example:

```bash
cargo run verify-proof root.txt 0 sample_tree_proof_0.json leaf.txt
```
