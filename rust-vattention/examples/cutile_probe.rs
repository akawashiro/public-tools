//! Verify that a cutile-generated cubin runs through the existing CUDA driver.
use anyhow::{ensure, Result};
use cudarc::{
    driver::{CudaContext, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};
fn main() -> Result<()> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = ctx.load_module(Ptx::from_binary(std::fs::read(
        "target/cutile/probe.cubin",
    )?))?;
    let function = module.load_function("probe_entry")?;
    let mut output = stream.alloc_zeros::<f32>(64)?;
    let n = 64i32;
    let one = 1i32;
    let mut launch = stream.launch_builder(&function);
    launch.arg(&mut output).arg(&n).arg(&one).arg(&n).arg(&one);
    // SAFETY: one output partition of 64 mapped floats, with matching shape/stride ABI.
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        })?;
    }
    let output = stream.clone_dtoh(&output)?;
    ensure!(
        output.iter().all(|x| *x == 7.),
        "incorrect cutile output: {output:?}"
    );
    println!("cutile cubin / cudarc driver interoperability passed: 64 elements");
    Ok(())
}
