use metal::{
    Buffer, CommandBufferRef, CommandQueue, ComputeCommandEncoderRef, ComputePipelineState, Device,
    Library, MTLResourceOptions,
};

use super::errors::MetalError;

use core::{ffi, mem};

const LIB_DATA: &[u8] = include_bytes!("../../../../math/src/gpu/metal/lib.metallib");

/// Structure for abstracting basic calls to a Metal device and saving the state. Used for
/// implementing GPU parallel computations in Apple machines.
pub struct MetalState {
    pub device: Device,
    pub library: Library,
    pub queue: CommandQueue,
}

impl MetalState {
    /// Creates a new Metal state with an optional `device` (GPU). If `None` is passed then it will use
    /// the system's default.
    pub fn new(device: Option<Device>) -> Result<Self, MetalError> {
        let device: Device =
            device.unwrap_or(Device::system_default().ok_or(MetalError::DeviceNotFound())?);

        let library = device
            .new_library_with_data(LIB_DATA) // TODO: allow different files
            .map_err(MetalError::LibraryError)?;
        let queue = device.new_command_queue();

        Ok(Self {
            device,
            library,
            queue,
        })
    }

    /// Creates a pipeline based on a compute function `kernel` which needs to exist in the state's
    /// library. A pipeline is used for issuing commands to the GPU through command buffers,
    /// executing the `kernel` function.
    pub fn setup_pipeline(&self, kernel_name: &str) -> Result<ComputePipelineState, MetalError> {
        let kernel = self
            .library
            .get_function(kernel_name, None)
            .map_err(MetalError::FunctionError)?;

        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(MetalError::PipelineError)?;

        Ok(pipeline)
    }

    /// Allocates `length` bytes of shared memory between CPU and the device (GPU).
    pub fn alloc_buffer<T>(&self, length: usize) -> Buffer {
        let size = mem::size_of::<T>();

        self.device.new_buffer(
            (length * size) as u64,
            MTLResourceOptions::StorageModeShared, // TODO: use managed mode
        )
    }

    /// Allocates `data` in a buffer of shared memory between CPU and the device (GPU).
    pub fn alloc_buffer_data<T>(&self, data: &[T]) -> Buffer {
        self.device.new_buffer_with_data(
            data.as_ptr() as *const ffi::c_void,
            std::mem::size_of_val(data) as u64,
            MTLResourceOptions::StorageModeShared, // TODO: use managed mode
        )
    }

    /// Creates a command buffer and a compute encoder in a pipeline, optionally issuing `buffers`
    /// to it.
    pub fn setup_command(
        &self,
        pipeline: &ComputePipelineState,
        buffers: Option<&[(u64, &Buffer)]>,
    ) -> (&CommandBufferRef, &ComputeCommandEncoderRef) {
        let command_buffer = self.queue.new_command_buffer();
        let command_encoder = command_buffer.new_compute_command_encoder();
        command_encoder.set_compute_pipeline_state(pipeline);

        if let Some(buffers) = buffers {
            for (i, buffer) in buffers.iter() {
                command_encoder.set_buffer(*i, Some(buffer), 0);
            }
        }

        (command_buffer, command_encoder)
    }

    /// Returns a vector of a copy of the data that `buffer` holds, interpreting it into a specific
    /// type `T`.
    ///
    /// BEWARE: this function uses an unsafe function for retrieveing the data, if the buffer's
    /// contents don't match the specified `T`, expect undefined behaviour. Always make sure the
    /// buffer you are retreiving from holds data of type `T`.
    pub fn retrieve_contents<T: Clone>(buffer: &Buffer) -> Vec<T> {
        let ptr = buffer.contents() as *const T;
        let len = buffer.length() as usize / mem::size_of::<T>();
        let slice = unsafe { std::slice::from_raw_parts(ptr, len) };

        slice.to_vec()
    }
}
