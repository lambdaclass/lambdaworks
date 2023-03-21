use metal::MTLResourceOptions;

use crate::abstractions::errors::MetalError;

use core::mem;

const LIB_DATA: &[u8] = include_bytes!("../metal/fft.metallib");

pub struct MetalState {
    pub device: metal::Device,
    pub library: metal::Library,
    pub queue: metal::CommandQueue,
}

impl MetalState {
    pub fn new(device: Option<metal::Device>) -> Result<Self, MetalError> {
        let device: metal::Device = device
            .unwrap_or(metal::Device::system_default().ok_or(MetalError::MetalDeviceNotFound())?);

        let library = device
            .new_library_with_data(LIB_DATA) // TODO: allow different files
            .map_err(MetalError::MetalLibraryError)?;
        let queue = device.new_command_queue();

        Ok(Self {
            device,
            library,
            queue,
        })
    }

    pub fn setup_pipeline(
        &self,
        kernel_name: &str,
    ) -> Result<metal::ComputePipelineState, MetalError> {
        let kernel = self
            .library
            .get_function(kernel_name, None)
            .map_err(MetalError::MetalFunctionError)?;

        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(MetalError::MetalPipelineError)?;

        Ok(pipeline)
    }

    pub fn alloc_buffer<T>(&self, length: usize) -> metal::Buffer {
        let size = mem::size_of::<T>();

        self.device.new_buffer(
            (length * size) as u64,
            MTLResourceOptions::StorageModeShared, // TODO: use managed mode
        )
    }

    pub fn alloc_buffer_data<T>(&self, data: &[T]) -> metal::Buffer {
        let size = mem::size_of::<T>();

        self.device.new_buffer_with_data(
            unsafe { mem::transmute(data.as_ptr()) },
            (data.len() * size) as u64,
            MTLResourceOptions::StorageModeShared, // TODO: use managed mode
        )
    }

    pub fn setup_command(
        &self,
        pipeline: &metal::ComputePipelineState,
        buffers: &[&metal::Buffer],
    ) -> (&metal::CommandBufferRef, &metal::ComputeCommandEncoderRef) {
        let command_buffer = self.queue.new_command_buffer();
        let command_encoder = command_buffer.new_compute_command_encoder();
        command_encoder.set_compute_pipeline_state(pipeline);

        for (i, buffer) in buffers.iter().enumerate() {
            command_encoder.set_buffer(i as u64, Some(buffer), 0);
        }

        (command_buffer, command_encoder)
    }

    pub fn retrieve_contents<T: Clone>(buffer: &metal::Buffer) -> Vec<T> {
        let ptr = buffer.contents() as *const T;
        let len = buffer.length() as usize / mem::size_of::<T>();
        let slice = unsafe { std::slice::from_raw_parts(ptr, len) };

        slice.to_vec()
    }
}
