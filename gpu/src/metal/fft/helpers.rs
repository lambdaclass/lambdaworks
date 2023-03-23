use lambdaworks_math::fft::errors::FFTError;

pub fn log2(n: usize) -> Result<u64, FFTError> {
    if !n.is_power_of_two() {
        return Err(FFTError::InvalidOrder(
            "The order of polynomial + 1 should a be power of 2".to_string(),
        ));
    }
    Ok(n.trailing_zeros() as u64)
}

pub fn void_ptr<T>(v: &T) -> *const core::ffi::c_void {
    v as *const T as *const core::ffi::c_void
}
