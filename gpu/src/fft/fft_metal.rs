use lambdaworks_math::fft::errors::FFTError;
use metal::{CommandQueue, Device, Library};

const FFT_LIB_DATA: &[u8] = include_bytes!("../metal/fft.metallib");

pub struct FFTMetalState {
    pub device: Device,
    pub library: Library,
    pub command_queue: CommandQueue,
}

impl FFTMetalState {
    pub fn new(device: Option<Device>) -> Result<Self, FFTError> {
        let device: Device =
            device.unwrap_or(Device::system_default().ok_or(FFTError::MetalDeviceNotFound())?);

        let library = device
            .new_library_with_data(FFT_LIB_DATA)
            .map_err(FFTError::MetalLibraryError)?;
        let command_queue = device.new_command_queue();

        Ok(FFTMetalState {
            device,
            library,
            command_queue,
        })
    }
}
