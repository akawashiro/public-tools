// Unpack a single QKV projection directly into Q and stable VMM cache addresses,
// combining optional Q/K head normalization with RoPE. V is copied unmodified.
extern "C" __global__ void qkv_transform(float* out_q, const float* qkv,
    float* keys, float* values, const W* qnorm, const W* knorm,
    U head, U heads, U kv_heads, U eps, U theta, U pos) {
    if (pos==~(U)0) pos=step_pos;
    U count=heads+2*kv_heads, token=blockIdx.x/count, h=blockIdx.x%count;
    const float* source=qkv+(token*count+h)*head;
    if (h>=heads+kv_heads) {
        float* dest=values+(pos+token)*kv_heads*head+(h-heads-kv_heads)*head;
        for (U i=threadIdx.x;i<head;i+=blockDim.x) dest[i]=source[i];
        return;
    }
    bool key=h>=heads;
    float* dest=key ? keys+(pos+token)*kv_heads*head+(h-heads)*head : out_q+(token*heads+h)*head;
    const W* norm=key ? knorm : qnorm;
    float scale=1.f;
    if (norm) {
        float sum=0;
        for (U i=threadIdx.x;i<head;i+=blockDim.x) sum+=source[i]*source[i];
        scale=rsqrtf(sum_block(sum)/float(head)+__uint_as_float((unsigned)eps));
    }
    for (U i=threadIdx.x*2;i<head;i+=blockDim.x*2) {
        float a=source[i]*scale*(norm ? load_weight(norm,i) : 1.f);
        float b=source[i+1]*scale*(norm ? load_weight(norm,i+1) : 1.f);
        float angle=float(pos+token)/powf(__uint_as_float((unsigned)theta),float(i)/float(head));
        float sn=sinf(angle),cs=cosf(angle);
        dest[i]=a*cs-b*sn; dest[i+1]=a*sn+b*cs;
    }
}
extern "C" __global__ void swiglu_pair(float* out, const float* gateup, U hidden, U count) {
    U i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=hidden*count) return;
    U token=i/hidden,j=i%hidden;
    float x=gateup[token*2*hidden+j];
    out[i]=x/(1.f+expf(-x))*gateup[token*2*hidden+hidden+j];
}
