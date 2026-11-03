//! CUDA VMM: stable virtual addresses, physical allocations added on demand.
use super::check;
use anyhow::{ensure, Context, Result};
use cudarc::driver::{sys, CudaContext};
use std::sync::Arc;

pub(crate) fn round_up(value: usize, granularity: usize) -> Result<usize> {
    ensure!(granularity > 0, "zero VMM granularity");
    value
        .checked_add(granularity - 1)
        .map(|n| n / granularity * granularity)
        .context("VMM size overflow")
}

struct Mapping {
    address: u64,
    size: usize,
    handle: sys::CUmemGenericAllocationHandle,
}

/// Not exposed as a host slice. Only GPU kernels can dereference this address.
pub(crate) struct VmmBuffer {
    ctx: Arc<CudaContext>,
    pub(crate) address: u64,
    pub(crate) reserved: usize,
    pub(crate) mapped: usize,
    chunk: usize,
    device: i32,
    mappings: Vec<Mapping>,
}

impl VmmBuffer {
    pub(crate) fn reserve(
        ctx: Arc<CudaContext>,
        bytes: usize,
        requested_chunk: usize,
    ) -> Result<Self> {
        ctx.bind_to_thread()?;
        ensure!(bytes > 0, "empty VMM reservation");
        ensure!(
            ctx.attribute(
                sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
            )? != 0,
            "GPU does not support CUDA virtual memory management"
        );
        // SAFETY: the CUDA ABI permits zero initialization of allocation props;
        // the required allocation type and location are set before use.
        let mut prop: sys::CUmemAllocationProp = unsafe { std::mem::zeroed() };
        prop.type_ = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = ctx.cu_device();
        let mut granularity = 0;
        // SAFETY: valid output pointer and initialized property struct; current context is bound.
        unsafe {
            check(
                sys::cuMemGetAllocationGranularity(
                    &mut granularity,
                    &prop,
                    sys::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM,
                ),
                "cuMemGetAllocationGranularity",
            )?;
        }
        let chunk = round_up(requested_chunk.max(granularity), granularity)?;
        let reserved = round_up(bytes, chunk)?;
        let mut address = 0;
        // SAFETY: CUDA owns the returned reservation; Drop frees it after all mappings.
        unsafe {
            check(
                sys::cuMemAddressReserve(&mut address, reserved, 0, 0, 0),
                "cuMemAddressReserve",
            )?;
        }
        Ok(Self {
            ctx,
            address,
            reserved,
            mapped: 0,
            chunk,
            device: prop.location.id,
            mappings: Vec::new(),
        })
    }

    pub(crate) fn ensure_bytes(&mut self, bytes: usize) -> Result<()> {
        ensure!(bytes <= self.reserved, "KV cache exceeds VMM reservation");
        self.ctx.bind_to_thread()?;
        let required = round_up(bytes, self.chunk)?;
        // SAFETY: CUDA allocation properties permit zero initialization; no host
        // pointers or platform handles are retained or moved between threads.
        let mut prop: sys::CUmemAllocationProp = unsafe { std::mem::zeroed() };
        prop.type_ = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type_ = sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = self.device;
        while self.mapped < required {
            let mut handle = 0;
            let address = self.address + self.mapped as u64;
            let access = sys::CUmemAccessDesc {
                location: prop.location,
                flags: sys::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
            };
            // SAFETY: each chunk is disjoint, aligned, and inside our reservation.
            // Failed mappings release their physical handle before returning.
            unsafe {
                check(
                    sys::cuMemCreate(&mut handle, self.chunk, &prop, 0),
                    "cuMemCreate",
                )?;
                if let Err(e) = check(sys::cuMemMap(address, self.chunk, 0, handle, 0), "cuMemMap")
                {
                    let _ = sys::cuMemRelease(handle);
                    return Err(e);
                }
                if let Err(e) = check(
                    sys::cuMemSetAccess(address, self.chunk, &access, 1),
                    "cuMemSetAccess",
                ) {
                    let _ = sys::cuMemUnmap(address, self.chunk);
                    let _ = sys::cuMemRelease(handle);
                    return Err(e);
                }
            }
            self.mappings.push(Mapping {
                address,
                size: self.chunk,
                handle,
            });
            self.mapped += self.chunk;
        }
        Ok(())
    }

    /// Caller must ensure that kernels using this cache have completed.
    pub(crate) fn release(&mut self) -> Result<()> {
        self.ctx.bind_to_thread()?;
        while let Some(mapping) = self.mappings.last() {
            // SAFETY: caller synchronizes; mapping is exclusively owned here.
            unsafe {
                check(sys::cuMemUnmap(mapping.address, mapping.size), "cuMemUnmap")?;
                check(sys::cuMemRelease(mapping.handle), "cuMemRelease")?;
            }
            self.mapped -= mapping.size;
            self.mappings.pop();
        }
        Ok(())
    }
}

impl Drop for VmmBuffer {
    fn drop(&mut self) {
        if self.ctx.synchronize().is_ok() && self.release().is_ok() {
            // SAFETY: all mappings have been removed; address reservation is ours.
            unsafe {
                let _ = sys::cuMemAddressFree(self.address, self.reserved);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn granularity_alignment() {
        assert_eq!(round_up(0, 65536).unwrap(), 0);
        assert_eq!(round_up(1, 65536).unwrap(), 65536);
        assert_eq!(round_up(65536, 65536).unwrap(), 65536);
        assert_eq!(round_up(65537, 65536).unwrap(), 131072);
        assert!(round_up(usize::MAX, 65536).is_err());
        assert!(round_up(1, 0).is_err());
    }

    #[test]
    #[ignore = "requires a CUDA GPU and driver"]
    fn cuda_mapping_growth_preserves_addresses_and_data() -> Result<()> {
        let ctx = CudaContext::new(0)?;
        let mut b = VmmBuffer::reserve(ctx.clone(), 32 * 1024 * 1024, 64 * 1024)?;
        let base = b.address;
        assert_eq!(b.mapped, 0);
        b.ensure_bytes(4)?;
        let chunk = b.mapped;
        let original = 1.2345f32;
        // SAFETY: the first mapped prefix spans this one-float copy.
        unsafe {
            check(
                sys::cuMemcpyHtoD_v2(base, (&original as *const f32).cast(), 4),
                "test upload",
            )?;
        }
        b.ensure_bytes(chunk + 4)?;
        assert_eq!(b.address, base);
        assert_eq!(b.mapped, 2 * chunk);
        let second = 6.789f32;
        let mut a = 0f32;
        let mut z = 0f32;
        // SAFETY: copies remain inside mapped allocations; transfers are synchronous.
        unsafe {
            check(
                sys::cuMemcpyHtoD_v2(base + chunk as u64, (&second as *const f32).cast(), 4),
                "test upload second chunk",
            )?;
            check(
                sys::cuMemcpyDtoH_v2((&mut a as *mut f32).cast(), base, 4),
                "test read first chunk",
            )?;
            check(
                sys::cuMemcpyDtoH_v2((&mut z as *mut f32).cast(), base + chunk as u64, 4),
                "test read second chunk",
            )?;
        }
        assert_eq!(a, original);
        assert_eq!(z, second);
        ctx.synchronize()?;
        b.release()?;
        assert_eq!(b.mapped, 0);
        assert_eq!(b.address, base);
        b.ensure_bytes(4)?;
        assert_eq!(b.mapped, chunk);
        assert!(b.ensure_bytes(b.reserved + 1).is_err());
        Ok(())
    }
}
