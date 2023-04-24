use cudarc::driver::safe::DeviceRepr;
use lambdaworks_math::field::{element::FieldElement, traits::IsField};

use core::ffi;

#[derive(Clone)]
pub struct CUDAFieldElement<F: IsField> {
    value: F::BaseType,
}

impl<F: IsField> CUDAFieldElement<F> {
    /// Returns the underlying `value`
    pub fn value(&self) -> &F::BaseType {
        &self.value
    }
}

impl<F: IsField> Default for CUDAFieldElement<F> {
    fn default() -> Self {
        Self { value: F::zero() }
    }
}

unsafe impl<F: IsField> DeviceRepr for CUDAFieldElement<F> {
    fn as_kernel_param(&self) -> *mut ffi::c_void {
        [self.value()].as_ptr() as *mut ffi::c_void
    }
}

impl<F: IsField> From<&FieldElement<F>> for CUDAFieldElement<F> {
    fn from(elem: &FieldElement<F>) -> Self {
        Self {
            value: elem.value().clone(),
        }
    }
}

impl<F: IsField> Into<FieldElement<F>> for CUDAFieldElement<F> {
    fn into(self) -> FieldElement<F> {
        FieldElement::from(self.value())
    }
}
