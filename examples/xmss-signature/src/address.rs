//! ADRS (Address) structure for domain separation in XMSS
//!
//! The 32-byte ADRS structure ensures that different hash function calls
//! within XMSS are domain-separated, preventing attacks that exploit
//! hash collisions across different contexts.

/// Address types for different XMSS operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum AddressType {
    /// OTS (One-Time Signature) hash address - used in WOTS+ chains
    Ots = 0,
    /// L-tree address - used when compressing WOTS+ public key
    LTree = 1,
    /// Hash tree address - used in XMSS Merkle tree
    HashTree = 2,
}

/// 32-byte address structure for domain separation
///
/// Layout (RFC 8391 Section 2.5):
/// - Bytes 0-3: Layer address (for XMSS^MT, 0 for single-tree)
/// - Bytes 4-11: Tree address (for XMSS^MT, 0 for single-tree)
/// - Bytes 12-15: Type (OTS=0, L-tree=1, Hash tree=2)
/// - Bytes 16-31: Type-specific fields
#[derive(Debug, Clone, Copy, Default)]
pub struct Address {
    data: [u8; 32],
}

impl Address {
    /// Create a new zeroed address
    pub fn new() -> Self {
        Self { data: [0u8; 32] }
    }

    /// Get the raw bytes of the address
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.data
    }

    /// Set layer address (bytes 0-3)
    pub fn set_layer(&mut self, layer: u32) {
        self.data[0..4].copy_from_slice(&layer.to_be_bytes());
    }

    /// Set tree address (bytes 4-11)
    pub fn set_tree(&mut self, tree: u64) {
        self.data[4..12].copy_from_slice(&tree.to_be_bytes());
    }

    /// Set address type (bytes 12-15)
    pub fn set_type(&mut self, addr_type: AddressType) {
        self.data[12..16].copy_from_slice(&(addr_type as u32).to_be_bytes());
    }

    /// Get the address type
    pub fn get_type(&self) -> u32 {
        u32::from_be_bytes([self.data[12], self.data[13], self.data[14], self.data[15]])
    }

    // OTS Address fields (when type = 0)

    /// Set OTS address - which WOTS+ key pair (bytes 16-19)
    pub fn set_ots_address(&mut self, ots: u32) {
        self.data[16..20].copy_from_slice(&ots.to_be_bytes());
    }

    /// Get OTS address
    pub fn get_ots_address(&self) -> u32 {
        u32::from_be_bytes([self.data[16], self.data[17], self.data[18], self.data[19]])
    }

    /// Set chain address - which chain within WOTS+ (bytes 20-23)
    pub fn set_chain_address(&mut self, chain: u32) {
        self.data[20..24].copy_from_slice(&chain.to_be_bytes());
    }

    /// Get chain address
    pub fn get_chain_address(&self) -> u32 {
        u32::from_be_bytes([self.data[20], self.data[21], self.data[22], self.data[23]])
    }

    /// Set hash address - position within chain (bytes 24-27)
    pub fn set_hash_address(&mut self, hash: u32) {
        self.data[24..28].copy_from_slice(&hash.to_be_bytes());
    }

    /// Get hash address
    pub fn get_hash_address(&self) -> u32 {
        u32::from_be_bytes([self.data[24], self.data[25], self.data[26], self.data[27]])
    }

    /// Set key and mask flag (bytes 28-31)
    /// 0 = key, 1 = bitmask first half, 2 = bitmask second half
    pub fn set_key_and_mask(&mut self, km: u32) {
        self.data[28..32].copy_from_slice(&km.to_be_bytes());
    }

    /// Get key and mask flag
    pub fn get_key_and_mask(&self) -> u32 {
        u32::from_be_bytes([self.data[28], self.data[29], self.data[30], self.data[31]])
    }

    // L-tree Address fields (when type = 1)

    /// Set L-tree address - which L-tree (bytes 16-19)
    pub fn set_ltree_address(&mut self, ltree: u32) {
        self.data[16..20].copy_from_slice(&ltree.to_be_bytes());
    }

    /// Get L-tree address
    pub fn get_ltree_address(&self) -> u32 {
        u32::from_be_bytes([self.data[16], self.data[17], self.data[18], self.data[19]])
    }

    // Tree fields (when type = 1 or 2) - shared between L-tree and Hash tree

    /// Set tree height (bytes 20-23) - height within tree
    pub fn set_tree_height(&mut self, height: u32) {
        self.data[20..24].copy_from_slice(&height.to_be_bytes());
    }

    /// Get tree height
    pub fn get_tree_height(&self) -> u32 {
        u32::from_be_bytes([self.data[20], self.data[21], self.data[22], self.data[23]])
    }

    /// Set tree index (bytes 24-27) - index within height level
    pub fn set_tree_index(&mut self, index: u32) {
        self.data[24..28].copy_from_slice(&index.to_be_bytes());
    }

    /// Get tree index
    pub fn get_tree_index(&self) -> u32 {
        u32::from_be_bytes([self.data[24], self.data[25], self.data[26], self.data[27]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_address_default() {
        let addr = Address::new();
        assert_eq!(addr.as_bytes(), &[0u8; 32]);
    }

    #[test]
    fn test_address_type() {
        let mut addr = Address::new();

        addr.set_type(AddressType::Ots);
        assert_eq!(addr.get_type(), 0);

        addr.set_type(AddressType::LTree);
        assert_eq!(addr.get_type(), 1);

        addr.set_type(AddressType::HashTree);
        assert_eq!(addr.get_type(), 2);
    }

    #[test]
    fn test_ots_address_fields() {
        let mut addr = Address::new();
        addr.set_type(AddressType::Ots);

        addr.set_ots_address(42);
        assert_eq!(addr.get_ots_address(), 42);

        addr.set_chain_address(15);
        assert_eq!(addr.get_chain_address(), 15);

        addr.set_hash_address(7);
        assert_eq!(addr.get_hash_address(), 7);

        addr.set_key_and_mask(1);
        assert_eq!(addr.get_key_and_mask(), 1);
    }

    #[test]
    fn test_tree_address_fields() {
        let mut addr = Address::new();
        addr.set_type(AddressType::HashTree);

        addr.set_tree_height(5);
        assert_eq!(addr.get_tree_height(), 5);

        addr.set_tree_index(10);
        assert_eq!(addr.get_tree_index(), 10);
    }

    #[test]
    fn test_layer_and_tree() {
        let mut addr = Address::new();

        addr.set_layer(3);
        addr.set_tree(0x123456789ABCDEF0);

        assert_eq!(&addr.data[0..4], &[0x00, 0x00, 0x00, 0x03]);
        assert_eq!(
            &addr.data[4..12],
            &[0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0]
        );
    }
}
