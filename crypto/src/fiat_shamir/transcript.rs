pub trait Transcript {
    fn append(&mut self, new_data: &[u8]);
    #[cfg(not(feature = "esp"))]
    fn challenge(&mut self) -> [u8; 32];
    #[cfg(feature = "esp")]
    fn challenge(&mut self) -> [u8; 28];
}
