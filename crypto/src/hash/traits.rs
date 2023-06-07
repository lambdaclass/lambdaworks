/// Interface to Collision Resistant Hashes.
pub trait IsHasher {
    type Type;
    type UnHashedLeaf;

    fn hash_leaf(&self, leaf: &Self::UnHashedLeaf) -> Self::Type;

    fn hash_leaves(&self, unhashed_leaves: &[Self::UnHashedLeaf]) -> Vec<Self::Type> {
        unhashed_leaves
            .iter()
            .map(|leaf| self.hash_leaf(leaf))
            .collect()
    }

    fn hash_two(&self, first: &Self::Type, second: &Self::Type) -> Self::Type;
}
