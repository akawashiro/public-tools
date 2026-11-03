extern "C" __global__ void causal_probabilities(const float* scores,float* probs,U length,
    U capacity,U queries,U query_start,U position) {
    U h=blockIdx.x/queries,q=blockIdx.x%queries;
    U base=(h*capacity+q)*length,allowed=position+query_start+q+1;
    U first=attention_first(allowed);
    float maximum=-__int_as_float(0x7f800000);
    for (U i=threadIdx.x;i<length;i+=256) if (i>=first && i<allowed) maximum=fmaxf(maximum,scores[base+i]);
    maximum=max_block(maximum);
    float sum=0;
    for (U i=threadIdx.x;i<length;i+=256) if (i>=first && i<allowed) sum+=expf(scores[base+i]-maximum);
    sum=sum_block(sum);
    for (U i=threadIdx.x;i<length;i+=256) probs[base+i]=i>=first && i<allowed ? expf(scores[base+i]-maximum)/sum : 0.f;
}
// The GEMM output is head-major. Write the following projection's FP16 input
// directly, avoiding a separate FP32 transpose and conversion buffer.
extern "C" __global__ void attention_output_half(unsigned short* out,const float* input,U head,U heads,U tokens) {
    U i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<tokens*heads*head) {
        U d=i%head,h=(i/head)%heads,t=i/(head*heads);
        float value=input[(h*tokens+t)*head+d];
        unsigned short half;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(half) : "f"(value));
        out[i]=half;
    }
}
