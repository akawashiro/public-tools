// Exact full-vocabulary nucleus sampling: stable radix sort and parallel CDFs.
__device__ float sample_temperature,sample_top_p,sample_uniform;
extern "C" __global__ void set_sampling(U temperature,U top_p,U uniform) {
    if (threadIdx.x==0) {
        sample_temperature=__uint_as_float((unsigned)temperature);
        sample_top_p=__uint_as_float((unsigned)top_p);
        sample_uniform=__uint_as_float((unsigned)uniform);
    }
}
extern "C" __global__ void sampling_init(const float* logits,const float* maximum,
    unsigned* keys,unsigned* ids,float* weights,float* partial,U count) {
    float sum=0, m=maximum[0];
    U end=min(count,(blockIdx.x+1)*(U)1024);
    for (U i=blockIdx.x*1024+threadIdx.x;i<end;i+=256) {
        float value=logits[i];
        float w=expf((float)((double(value)-double(m))/double(sample_temperature)));
        weights[i]=w; sum+=w;
    }
    sum=sum_block(sum);
    if (threadIdx.x==0) partial[blockIdx.x]=sum;
}
extern "C" __global__ void sampling_sum(const float* partial,float* total,U count) {
    float value=0;
    for (U i=threadIdx.x;i<count;i+=256) value+=partial[i];
    value=sum_block(value);
    if (threadIdx.x==0) total[0]=value;
}
extern "C" __global__ void sampling_keys(float* weights,unsigned* keys,unsigned* ids,const float* total,U count) {
    U i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<count) {
        float p=weights[i]/total[0]; weights[i]=p;
        keys[i]=~__float_as_uint(p); ids[i]=(unsigned)i;
    }
}
extern "C" __global__ void radix_hist(const unsigned* keys,unsigned* hist,U count,U shift) {
    __shared__ unsigned counts[16];
    if (threadIdx.x<16) counts[threadIdx.x]=0;
    __syncthreads();
    U end=min(count,(blockIdx.x+1)*(U)1024);
    for (U i=blockIdx.x*1024+threadIdx.x;i<end;i+=256) atomicAdd(counts+((keys[i]>>shift)&15),1);
    __syncthreads();
    if (threadIdx.x<16) hist[blockIdx.x*16+threadIdx.x]=counts[threadIdx.x];
}
extern "C" __global__ void radix_offsets(const unsigned* hist,unsigned* offsets,U blocks) {
    U lane=threadIdx.x%32,warp=threadIdx.x/32;
    __shared__ unsigned totals[16];
    for (U bin=warp;bin<16;bin+=8) {
        unsigned carry=0;
        for (U start=0;start<blocks;start+=32) {
            U i=start+lane; unsigned value=i<blocks ? hist[i*16+bin] : 0;
            unsigned inclusive=value;
            for (int d=1;d<32;d*=2) { unsigned x=__shfl_up_sync(0xffffffff,inclusive,d); if (lane>=(U)d) inclusive+=x; }
            if (i<blocks) offsets[bin*blocks+i]=carry+inclusive-value;
            carry+=__shfl_sync(0xffffffff,inclusive,31);
        }
        if (lane==0) totals[bin]=carry;
    }
    __syncthreads();
    for (U i=threadIdx.x;i<16*blocks;i+=256) {
        U bin=i/blocks; unsigned base=0;
        for (U b=0;b<bin;++b) base+=totals[b];
        offsets[i]+=base;
    }
}
extern "C" __global__ void radix_scatter(const unsigned* in_keys,const unsigned* in_ids,
    unsigned* out_keys,unsigned* out_ids,const unsigned* offsets,U count,U blocks,U shift) {
    U lane=threadIdx.x%32,warp=threadIdx.x/32;
    __shared__ unsigned counts[512],prefix[512];
    for (U i=threadIdx.x;i<512;i+=256) counts[i]=0;
    __syncthreads();
    unsigned saved_keys[4],ranks[4];
#pragma unroll
    for (unsigned j=0;j<4;++j) {
        U i=blockIdx.x*1024+j*256+threadIdx.x;
        unsigned key=i<count ? in_keys[i] : 0;
        unsigned bin=(key>>shift)&15;
        unsigned matches=__ballot_sync(0xffffffff,i<count);
        for (unsigned bit=0;bit<4;++bit) {
            unsigned mask=__ballot_sync(0xffffffff,(bin>>bit)&1);
            matches&=((bin>>bit)&1) ? mask : ~mask;
        }
        saved_keys[j]=key;
        ranks[j]=__popc(matches&((1u<<lane)-1));
        if (i<count && lane==(unsigned)(__ffs(matches)-1)) counts[(j*8+warp)*16+bin]=__popc(matches);
    }
    __syncthreads();
    for (U bin=warp;bin<16;bin+=8) {
        unsigned value=counts[lane*16+bin],inclusive=value;
        for (int d=1;d<32;d*=2) { unsigned x=__shfl_up_sync(0xffffffff,inclusive,d); if (lane>=(U)d) inclusive+=x; }
        prefix[bin*32+lane]=inclusive-value;
    }
    __syncthreads();
#pragma unroll
    for (unsigned j=0;j<4;++j) {
        U i=blockIdx.x*1024+j*256+threadIdx.x;
        if (i<count) {
            unsigned key=saved_keys[j],bin=(key>>shift)&15;
            unsigned target=offsets[bin*blocks+blockIdx.x]+prefix[bin*32+j*8+warp]+ranks[j];
            out_keys[target]=key; out_ids[target]=in_ids[i];
        }
    }
}
struct FloatScan { float inclusive,total; };
__device__ FloatScan scan_float(float value) {
    U lane=threadIdx.x%32,warp=threadIdx.x/32;
    __shared__ float totals[8];
    for (int d=1;d<32;d*=2) { float x=__shfl_up_sync(0xffffffff,value,d); if (lane>=(U)d) value+=x; }
    if (lane==31) totals[warp]=value;
    __syncthreads();
    float base=0,total=0;
    for (U w=0;w<8;++w) { if (w<warp) base+=totals[w]; total+=totals[w]; }
    __syncthreads();
    return {value+base,total};
}
extern "C" __global__ void ordered_mass(const float* weights,const unsigned* ids,
    const float* total,float* masses,U count) {
    float sum=0; U end=min(count,(blockIdx.x+1)*(U)1024);
    for (U i=blockIdx.x*1024+threadIdx.x;i<end;i+=256) sum+=weights[ids[i]];
    sum=sum_block(sum);
    if (threadIdx.x==0) masses[blockIdx.x]=sum;
}
extern "C" __global__ void mass_prefix(const float* masses,float* prefix,U count) {
    float carry=0;
    for (U start=0;start<count;start+=256) {
        U i=start+threadIdx.x;
        FloatScan scan=scan_float(i<count ? masses[i] : 0.f);
        if (i<count) prefix[i]=carry+scan.inclusive;
        carry+=scan.total;
    }
}
__device__ void sample_block_prefix(float* prefix,const float* weights,const unsigned* ids,
    const float* total,U start,U count) {
    U index=start+threadIdx.x*4;
    float a=index<count ? weights[ids[index]] : 0.f;
    float b=index+1<count ? weights[ids[index+1]] : 0.f;
    float c=index+2<count ? weights[ids[index+2]] : 0.f;
    float d=index+3<count ? weights[ids[index+3]] : 0.f;
    float sum=((a+b)+c)+d;
    FloatScan scan=scan_float(sum);
    float base=scan.inclusive-sum;
    U i=threadIdx.x*4;
    prefix[i]=base+a; prefix[i+1]=(base+a)+b;
    prefix[i+2]=((base+a)+b)+c; prefix[i+3]=((base+a)+b)+c+d;
    __syncthreads();
}
extern "C" __global__ void sample_final(const float* weights,const unsigned* ids,const float* total,
    const float* mass_prefixes,unsigned* token,U count,U blocks) {
    if (token[0]==0xffffffff || !isfinite(total[0]) || total[0]<=0.f) {
        if (threadIdx.x==0) token[0]=0xffffffff;
        return;
    }
    __shared__ float prefix[1024],target;
    __shared__ unsigned nucleus_block,sample_block,end_index,chosen;
    if (threadIdx.x==0) { nucleus_block=blocks-1; sample_block=blocks-1; }
    __syncthreads();
    for (U i=threadIdx.x;i<blocks;i+=256) if (sample_top_p<1.f && mass_prefixes[i]>=sample_top_p) atomicMin(&nucleus_block,(unsigned)i);
    __syncthreads();
    U start=nucleus_block*1024;
    float before=nucleus_block ? mass_prefixes[nucleus_block-1] : 0.f;
    if (threadIdx.x==0) end_index=(unsigned)min((U)1024,count-start)-1;
    sample_block_prefix(prefix,weights,ids,total,start,count);
    for (U j=threadIdx.x;j<1024 && start+j<count;j+=256)
        if (sample_top_p<1.f && before+prefix[j]>=sample_top_p) atomicMin(&end_index,(unsigned)j);
    __syncthreads();
    if (threadIdx.x==0) target=sample_uniform*(before+prefix[end_index]);
    __syncthreads();
    for (U i=threadIdx.x;i<blocks;i+=256) if (mass_prefixes[i]>target) atomicMin(&sample_block,(unsigned)i);
    __syncthreads();
    start=sample_block*1024;
    before=sample_block ? mass_prefixes[sample_block-1] : 0.f;
    if (threadIdx.x==0) chosen=(unsigned)min((U)1024,count-start)-1;
    sample_block_prefix(prefix,weights,ids,total,start,count);
    for (U j=threadIdx.x;j<1024 && start+j<count;j+=256)
        if (before+prefix[j]>target) atomicMin(&chosen,(unsigned)j);
    __syncthreads();
    if (threadIdx.x==0) token[0]=ids[start+chosen];
}
