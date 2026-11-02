// Ordinary dense attention: all KV addresses are base + token * stride.
// There is no application block table and no paged-attention indirection.
typedef unsigned long long U;
extern "C" __global__ void embedding(float* out, const float* w, U token, U d) {
    U i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < d) out[i] = w[token*d+i];
}
__device__ float sum_block(float x) {
    __shared__ float tmp[256];
    tmp[threadIdx.x] = x; __syncthreads();
    for (int step=128; step; step/=2) {
        if (threadIdx.x < step) tmp[threadIdx.x] += tmp[threadIdx.x+step];
        __syncthreads();
    }
    return tmp[0];
}
extern "C" __global__ void matvec(float* out, const float* x, const float* w, U n) {
    U row = blockIdx.x;
    float sum = 0;
    for (U i=threadIdx.x; i<n; i+=blockDim.x) sum += w[row*n+i]*x[i];
    sum = sum_block(sum);
    if (threadIdx.x==0) out[row] = sum;
}
extern "C" __global__ void rmsnorm(float* out, const float* x, const float* w, U n) {
    float sum = 0;
    for (U i=threadIdx.x; i<n; i+=blockDim.x) sum += x[i]*x[i];
    float scale = rsqrtf(sum_block(sum)/float(n)+1e-5f);
    for (U i=threadIdx.x; i<n; i+=blockDim.x) out[i] = x[i]*scale*w[i];
}
extern "C" __global__ void rope(float* x, U n, U head, U pos) {
    U i = (blockIdx.x*blockDim.x+threadIdx.x)*2;
    if (i>=n) return;
    float angle = float(pos)/powf(10000.f, float(i%head)/float(head));
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
    for (U t=0; t<length; ++t) sum += scores[h*length+t]*values[t*kv+offset];
    out[i] = sum;
}
