#[cfg(test)]
mod tests {
    use cudarc::driver::LaunchAsync;

    #[test]
    fn poc() {
        let dev = cudarc::driver::CudaDevice::new(0).unwrap();

        // allocate buffers
        let inp = dev.htod_copy(vec![1i32; 100]).unwrap();
        let mut out = dev.alloc_zeros::<i32>(100).unwrap();

        let ptx = cudarc::nvrtc::compile_ptx(
            "
        extern \"C\" __global__ void test_kernel(int *out, const int *inp, const size_t numel) {
            unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < numel) {
                out[i] = inp[i] * i;
            }
        }",
        )
        .unwrap();

        // and dynamically load it into the device
        dev.load_ptx(ptx, "my_module", &["test_kernel"]).unwrap();

        let test_kernel = dev.get_func("my_module", "test_kernel").unwrap();
        let cfg = cudarc::driver::LaunchConfig::for_num_elems(100);
        unsafe { test_kernel.launch(cfg, (&mut out, &inp, 100usize)) }.unwrap();

        let out_host: Vec<i32> = dev.dtoh_sync_copy(&out).unwrap();
        assert_eq!(out_host, (0..100).collect::<Vec<i32>>());
    }
}
