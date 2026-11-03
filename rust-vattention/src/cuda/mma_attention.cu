#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ unsigned tf32_bits(float value) {
    unsigned bits;asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(bits) : "f"(value));return bits;
}
__device__ __forceinline__ void mma_tf32(float* d,unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
#endif
// 16 queries x 32 keys per CTA; QK and PV stay in shared memory/registers.
// TF32 inputs, FP32 accumulation, FP16 projection output. FP32 VMM K/V are
// read directly with their token stride; no score matrix or KV copy exists.
extern "C" __global__ void mma_causal_attention(unsigned short* out,const float* query,
    const float* keys,const float* values,U tokens,U heads,U kv_heads,U position) {
#if __CUDA_ARCH__ >= 800
    const U lane=threadIdx.x%32,warp=threadIdx.x/32,group=lane/4,thread=lane%4;
    const U h=blockIdx.x%heads,qstart=(blockIdx.x/heads)*16,kh=h/(heads/kv_heads);
    const U end=position+min(qstart+16,tokens),kvstride=kv_heads*128;
    // Padding removes bank conflicts in the m16n8k8 register-fragment loads.
    __shared__ unsigned q[16*132],k[32*132],v[32*136],p[16*36];
    __shared__ float maximum[16],denom[16],alpha[16];
    for (U i=threadIdx.x;i<16*128;i+=128) {
        U row=i/128,col=i%128;
        float value=qstart+row<tokens ? query[((qstart+row)*heads+h)*128+col] : 0.f;
        q[row*132+col]=tf32_bits(value);
    }
    if (threadIdx.x<16) {maximum[threadIdx.x]=-__int_as_float(0x7f800000);denom[threadIdx.x]=0.f;}
    float acc[4][4]={{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};
    for (U start=0;start<end;start+=32) {
        for (U i=threadIdx.x;i<32*128;i+=128) {
            U row=i/128,col=i%128,t=start+row;
            k[row*132+col]=tf32_bits(t<end ? keys[t*kvstride+kh*128+col] : 0.f);
            v[row*136+col]=tf32_bits(t<end ? values[t*kvstride+kh*128+col] : 0.f);
        }
        __syncthreads();
        float scores[4]={0,0,0,0};
#pragma unroll
        for (unsigned dim=0;dim<128;dim+=8) {
            mma_tf32(scores,q[group*132+dim+thread],q[(group+8)*132+dim+thread],
                q[group*132+dim+thread+4],q[(group+8)*132+dim+thread+4],
                k[(warp*8+group)*132+dim+thread],k[(warp*8+group)*132+dim+thread+4]);
        }
        // Reuse p's storage for scores before replacing them by TF32 probabilities.
#pragma unroll
        for (unsigned j=0;j<4;++j) {
            U row=group+(j/2)*8,col=warp*8+thread*2+j%2;
            p[row*36+col]=__float_as_uint(scores[j]*0.08838834764831844f);
        }
        __syncthreads();
#pragma unroll
        for (unsigned row=warp;row<16;row+=4) {
            bool valid=qstart+row<tokens && start+lane<=position+qstart+row && start+lane<end;
            float score=valid ? __uint_as_float(p[row*36+lane]) : -__int_as_float(0x7f800000);
            float tile_max=score;
            for (int d=16;d;d/=2) tile_max=fmaxf(tile_max,__shfl_down_sync(0xffffffff,tile_max,d));
            tile_max=__shfl_sync(0xffffffff,tile_max,0);
            float next_max=fmaxf(maximum[row],tile_max);
            float scale=qstart+row<tokens ? expf(maximum[row]-next_max) : 0.f;
            float prob=valid ? expf(score-next_max) : 0.f;
            float sum=prob;
            for (int d=16;d;d/=2) sum+=__shfl_down_sync(0xffffffff,sum,d);
            if (lane==0) {maximum[row]=next_max;denom[row]=denom[row]*scale+sum;alpha[row]=scale;}
            p[row*36+lane]=tf32_bits(prob);
        }
        __syncthreads();
#pragma unroll
        for (unsigned j=0;j<4;++j) {
            acc[j][0]*=alpha[group];acc[j][1]*=alpha[group];
            acc[j][2]*=alpha[group+8];acc[j][3]*=alpha[group+8];
        }
#pragma unroll
        for (unsigned tile=0;tile<32;tile+=8) {
            unsigned a0=p[group*36+tile+thread],a1=p[(group+8)*36+tile+thread];
            unsigned a2=p[group*36+tile+thread+4],a3=p[(group+8)*36+tile+thread+4];
#pragma unroll
            for (unsigned j=0;j<4;++j) {
                U col=warp*32+j*8+group;
                mma_tf32(acc[j],a0,a1,a2,a3,v[(tile+thread)*136+col],v[(tile+thread+4)*136+col]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned j=0;j<4;++j) {
#pragma unroll
        for (unsigned i=0;i<4;++i) {
            U row=group+(i/2)*8,col=warp*32+j*8+thread*2+i%2;
            if (qstart+row<tokens) {
                float value=acc[j][i]/denom[row];unsigned short half;
                asm("cvt.rn.f16.f32 %0, %1;" : "=h"(half) : "f"(value));
                out[((qstart+row)*heads+h)*128+col]=half;
            }
        }
    }
#endif
}
