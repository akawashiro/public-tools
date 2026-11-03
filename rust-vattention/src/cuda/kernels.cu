// Ordinary dense attention: all KV addresses are base + token * stride.
// There is no application block table and no paged-attention indirection.
typedef unsigned long long U;
__device__ U step_token, step_pos;
extern "C" __global__ void set_step(U token, U pos) {
    if (threadIdx.x==0) { step_token=token; step_pos=pos; }
}

#ifdef WEIGHTS_FP16
typedef unsigned short W;
__device__ float load_weight(const W* w, U i) {
    float out;
    asm("cvt.f32.f16 %0, %1;" : "=f"(out) : "h"(w[i]));
    return out;
}
#else
typedef float W;
__device__ float load_weight(const W* w, U i) { return w[i]; }
#endif
extern "C" __global__ void embedding(float* out, const W* w, U token, U d) {
    U i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d) out[i] = load_weight(w, token*d+i);
}
__device__ float sum_block(float x) {
    __shared__ float warps[8];
    U lane=threadIdx.x%32, warp=threadIdx.x/32;
    for (int d=16;d;d/=2) x+=__shfl_down_sync(0xffffffff,x,d);
    if (lane==0) warps[warp]=x;
    __syncthreads();
    if (warp==0) {
        x=lane<8 ? warps[lane] : 0.f;
        for (int d=16;d;d/=2) x+=__shfl_down_sync(0xffffffff,x,d);
        if (lane==0) warps[0]=x;
    }
    __syncthreads();
    return warps[0];
}
// Eight independent rows per CTA. Each warp streams four contiguous weights
// per lane and reduces with shuffles, avoiding shared-memory CTA barriers.
extern "C" __global__ void matvec(float* out, const float* x, const W* w, U n, U rows) {
    U lane=threadIdx.x%32, row=blockIdx.x*8+threadIdx.x/32;
    if (row>=rows) return;
    float sum=0;
    if (n%4==0) {
        for (U i=lane*4;i<n;i+=128) {
            float4 v=*reinterpret_cast<const float4*>(x+i);
#ifdef WEIGHTS_FP16
            uint2 bits=*reinterpret_cast<const uint2*>(w+row*n+i);
            unsigned short h0=bits.x, h1=bits.x>>16, h2=bits.y, h3=bits.y>>16;
            float a,b,c,d;
            asm("cvt.f32.f16 %0, %1;" : "=f"(a) : "h"(h0));
            asm("cvt.f32.f16 %0, %1;" : "=f"(b) : "h"(h1));
            asm("cvt.f32.f16 %0, %1;" : "=f"(c) : "h"(h2));
            asm("cvt.f32.f16 %0, %1;" : "=f"(d) : "h"(h3));
#else
            float4 weight=*reinterpret_cast<const float4*>(w+row*n+i);
            float a=weight.x,b=weight.y,c=weight.z,d=weight.w;
#endif
            sum=__fmaf_rn(a,v.x,sum); sum=__fmaf_rn(b,v.y,sum);
            sum=__fmaf_rn(c,v.z,sum); sum=__fmaf_rn(d,v.w,sum);
        }
    } else {
        for (U i=lane;i<n;i+=32) sum=__fmaf_rn(load_weight(w,row*n+i),x[i],sum);
    }
    for (int d=16;d;d/=2) sum+=__shfl_down_sync(0xffffffff,sum,d);
    if (lane==0) out[row]=sum;
}
extern "C" __global__ void rmsnorm(float* out, const float* x, const W* w, U n, U eps) {
    float sum = 0;
    for (U i=threadIdx.x; i<n; i+=blockDim.x) sum += x[i]*x[i];
    float scale = rsqrtf(sum_block(sum)/float(n)+__uint_as_float((unsigned)eps));
    for (U i=threadIdx.x; i<n; i+=blockDim.x) out[i] = x[i]*scale*load_weight(w, i);
}
extern "C" __global__ void head_rmsnorm(float* x, const W* w, U head, U eps) {
    x += blockIdx.x * head;
    float sum = 0;
    for (U i=threadIdx.x; i<head; i+=blockDim.x) sum += x[i]*x[i];
    float scale = rsqrtf(sum_block(sum)/float(head)+__uint_as_float((unsigned)eps));
    for (U i=threadIdx.x; i<head; i+=blockDim.x) x[i] *= scale*load_weight(w, i);
}
extern "C" __global__ void rope(float* x, U n, U head, U pos, U theta) {
    U i = (blockIdx.x*blockDim.x+threadIdx.x)*2;
    if (i>=n) return;
    float angle = rope_angle(i,head,pos,__uint_as_float((unsigned)theta));
    float s = sinf(angle), c = cosf(angle), a = x[i], b = x[i+1];
    x[i] = a*c-b*s; x[i+1] = a*s+b*c;
}
extern "C" __global__ void add(float* x, const float* residual, U n) {
    U i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) x[i] += residual[i];
}
extern "C" __global__ void swiglu(float* h, const float* gate, U n) {
    U i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) h[i] = h[i]/(1.f+expf(-h[i]))*gate[i];
}
extern "C" __global__ void scores(float* scores, const float* q, const float* keys,
                                  U head, U kv, U kv_mul, U length) {
    U h = blockIdx.x/length, t = blockIdx.x%length;
    if (t<attention_first(length)) { if (threadIdx.x==0) scores[h*length+t]=-__int_as_float(0x7f800000); return; }
    U key = t*kv+(h/kv_mul)*head;
    float sum = 0;
    for (U i=threadIdx.x; i<head; i+=blockDim.x) sum += q[h*head+i]*keys[key+i];
    sum = sum_block(sum);
    if (threadIdx.x==0) scores[h*length+t] = sum/sqrtf(float(head));
}
extern "C" __global__ void attention_softmax(float* scores, U length) {
    float* x = scores+blockIdx.x*length;
    __shared__ float tmp[256];
    float m = -__int_as_float(0x7f800000);
    for (U t=threadIdx.x; t<length; t+=blockDim.x) m = fmaxf(m,x[t]);
    tmp[threadIdx.x] = m; __syncthreads();
    for (int step=128; step; step/=2) {
        if (threadIdx.x<step) tmp[threadIdx.x] = fmaxf(tmp[threadIdx.x],tmp[threadIdx.x+step]);
        __syncthreads();
    }
    m = tmp[0]; __syncthreads();
    float sum = 0;
    for (U t=threadIdx.x; t<length; t+=blockDim.x) { x[t] = expf(x[t]-m); sum += x[t]; }
    sum = sum_block(sum);
    for (U t=threadIdx.x; t<length; t+=blockDim.x) x[t] /= sum;
}
extern "C" __global__ void attention_values(float* out, const float* scores, const float* values,
                                            U dim, U head, U kv, U kv_mul, U length) {
    U i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=dim) return;
    U h = i/head, j = i%head, offset = (h/kv_mul)*head+j;
    float sum = 0;
    for (U t=attention_first(length); t<length; ++t) sum += scores[h*length+t]*values[t*kv+offset];
    out[i] = sum;
}

extern "C" __global__ void embedding_batch(float* out, const W* w, const U* ids, U d, U count) {
    U i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < count*d) out[i] = load_weight(w, ids[i/d]*d+i%d);
}
extern "C" __global__ void norm_batch(float* out, const float* x, const W* w, U d, U eps) {
    x += blockIdx.x*d; out += blockIdx.x*d;
    float sum = 0;
    for (U i=threadIdx.x; i<d; i+=blockDim.x) sum += x[i]*x[i];
    float scale = rsqrtf(sum_block(sum)/float(d)+__uint_as_float((unsigned)eps));
    for (U i=threadIdx.x; i<d; i+=blockDim.x) out[i] = x[i]*scale*load_weight(w,i);
}
extern "C" __global__ void to_half(unsigned short* out, const float* x, U n) {
    U i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) { unsigned short h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(x[i])); out[i] = h; }
}
__device__ float max_block(float x) {
    __shared__ float warps[8];
    U lane=threadIdx.x%32, warp=threadIdx.x/32;
    for (int d=16;d;d/=2) x=fmaxf(x,__shfl_down_sync(0xffffffff,x,d));
    if (lane==0) warps[warp]=x;
    __syncthreads();
    if (warp==0) {
        x=lane<8 ? warps[lane] : -__int_as_float(0x7f800000);
        for (int d=16;d;d/=2) x=fmaxf(x,__shfl_down_sync(0xffffffff,x,d));
        if (lane==0) warps[0]=x;
    }
    __syncthreads();
    return warps[0];
}
// Exact causal attention, online softmax in 256-key tiles, no score matrix.
// Every K/V access is into base + token*stride of a VMM reservation.
extern "C" __global__ void batch_attention(float* out, const float* q, const float* keys,
    const float* values, U head, U heads, U kv_heads, U pos) {
    if (pos==~(U)0) pos=step_pos;
    U h=blockIdx.x%heads, query=blockIdx.x/heads, length=pos+query+1;
    U kv=kv_heads*head, kh=(h/(heads/kv_heads))*head;
    U lane=threadIdx.x%32, warp=threadIdx.x/32;
    __shared__ float query_vec[256];
    extern __shared__ float scores_tile[];
    if (threadIdx.x<head) query_vec[threadIdx.x]=q[(query*heads+h)*head+threadIdx.x];
    __syncthreads();
    float m=-__int_as_float(0x7f800000), denom=0, acc=0;
    for (U start=attention_first(length); start<length; start+=256) {
        U count=min((U)256,length-start);
        for (U j=warp;j<256;j+=8) {
            float dot=0;
            if (j<count) for (U i=lane;i<head;i+=32) dot+=query_vec[i]*keys[(start+j)*kv+kh+i];
            for (int d=16;d;d/=2) dot+=__shfl_down_sync(0xffffffff,dot,d);
            if (lane==0) scores_tile[j]=j<count ? dot*rsqrtf(float(head)) : -__int_as_float(0x7f800000);
        }
        __syncthreads();
        float next_m=fmaxf(m,max_block(scores_tile[threadIdx.x]));
        float alpha=expf(m-next_m);
        float prob=expf(scores_tile[threadIdx.x]-next_m);
        float next_denom=denom*alpha+sum_block(prob);
        scores_tile[threadIdx.x]=prob;
        __syncthreads();
        if (threadIdx.x<head) {
            acc*=alpha;
            for (U j=0;j<count;++j) acc+=scores_tile[j]*values[(start+j)*kv+kh+threadIdx.x];
        }
        __syncthreads();
        m=next_m; denom=next_denom;
    }
    if (threadIdx.x<head) out[(query*heads+h)*head+threadIdx.x]=acc/denom;
}

extern "C" __global__ void embedding_step(float* out, const W* w, U d) {
    U i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<d) out[i]=load_weight(w,step_token*d+i);
}
// Residual addition and RMSNorm share the same read of the activation vector.
extern "C" __global__ void add_norm(float* normalized, float* x, const float* residual, const W* w, U n, U eps) {
    U token=blockIdx.x;
    x+=token*n; residual+=token*n; normalized+=token*n;
    float sum=0;
    for (U i=threadIdx.x;i<n;i+=blockDim.x) { float v=x[i]+residual[i]; x[i]=v; sum+=v*v; }
    float scale=rsqrtf(sum_block(sum)/float(n)+__uint_as_float((unsigned)eps));
    for (U i=threadIdx.x;i<n;i+=blockDim.x) normalized[i]=x[i]*scale*load_weight(w,i);
}

// Stable greedy reduction; equal logits choose the lowest token ID. Invalid
// floating-point values propagate a sentinel instead of being silently ignored.
extern "C" __global__ void argmax(const float* values, const unsigned* ids, U count,
    U span, float* out_values, unsigned* out_ids) {
    U lane=threadIdx.x%32, warp=threadIdx.x/32;
    float best=-__int_as_float(0x7f800000); unsigned id=0xffffffff, bad=0;
    U end=min(count,(blockIdx.x+1)*span);
    for (U i=blockIdx.x*span+threadIdx.x;i<end;i+=256) {
        float v=values[i]; unsigned index=ids ? ids[i] : (unsigned)i;
        bad|=!isfinite(v) || index==0xffffffff;
        if (v>best || (v==best && index<id)) { best=v; id=index; }
    }
    for (int d=16;d;d/=2) {
        float v=__shfl_down_sync(0xffffffff,best,d);
        unsigned index=__shfl_down_sync(0xffffffff,id,d);
        bad|=__shfl_down_sync(0xffffffff,bad,d);
        if (v>best || (v==best && index<id)) { best=v; id=index; }
    }
    __shared__ float maxima[8]; __shared__ unsigned indices[8],invalid[8];
    if (lane==0) { maxima[warp]=best; indices[warp]=id; invalid[warp]=bad; }
    __syncthreads();
    if (warp==0) {
        best=lane<8 ? maxima[lane] : -__int_as_float(0x7f800000);
        id=lane<8 ? indices[lane] : 0xffffffff; bad=lane<8 ? invalid[lane] : 0;
        for (int d=16;d;d/=2) {
            float v=__shfl_down_sync(0xffffffff,best,d);
            unsigned index=__shfl_down_sync(0xffffffff,id,d);
            bad|=__shfl_down_sync(0xffffffff,bad,d);
            if (v>best || (v==best && index<id)) { best=v; id=index; }
        }
        if (lane==0) { out_values[blockIdx.x]=best; out_ids[blockIdx.x]=bad ? 0xffffffff : id; }
    }
}

extern "C" __global__ void add_bias(float* x, const W* bias, U n) {
    U i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) x[i]+=load_weight(bias,i);
}
