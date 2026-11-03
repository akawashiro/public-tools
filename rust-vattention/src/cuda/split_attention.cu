// Parallel KV partitions are computation tiles, not allocation/block tables.
// Every address remains VMM base + token*KV_stride + head_offset.
extern "C" __global__ void decode_attention_partial(float* partial,const float* q,const float* keys,
    const float* values,U head,U heads,U kv_heads,U capacity) {
    U h=blockIdx.x%heads,segment=blockIdx.x/heads,start=segment*256,length=step_pos+1;
    U first=attention_first(length);
    if (start>=length || start+256<=first) return;
    U count=min((U)256,length-start),kv=kv_heads*head,kh=(h/(heads/kv_heads))*head;
    U lane=threadIdx.x%32,warp=threadIdx.x/32;
    __shared__ float query[256],scores[256];
    if (threadIdx.x<head) query[threadIdx.x]=q[h*head+threadIdx.x];
    __syncthreads();
    for (U j=warp;j<256;j+=8) {
        float dot=0;
        if (j<count && start+j>=first) for (U i=lane;i<head;i+=32) dot+=query[i]*keys[(start+j)*kv+kh+i];
        for (int d=16;d;d/=2) dot+=__shfl_down_sync(0xffffffff,dot,d);
        if (lane==0) scores[j]=j<count && start+j>=first ? dot*rsqrtf(float(head)) : -__int_as_float(0x7f800000);
    }
    __syncthreads();
    float maximum=max_block(scores[threadIdx.x]);
    float p=expf(scores[threadIdx.x]-maximum);
    float denom=sum_block(p);
    scores[threadIdx.x]=p;
    __syncthreads();
    U base=(h*capacity+segment)*(head+2);
    if (threadIdx.x<head) {
        float acc=0;
        for (U j=0;j<count;++j) acc=__fmaf_rn(scores[j],values[(start+j)*kv+kh+threadIdx.x],acc);
        partial[base+threadIdx.x]=acc;
    }
    if (threadIdx.x==0) {partial[base+head]=maximum;partial[base+head+1]=denom;}
}
extern "C" __global__ void decode_attention_merge(float* out,const float* partial,U head,U capacity) {
    U h=blockIdx.x,count=(step_pos+256)/256,first=attention_first(step_pos+1)/256;
    extern __shared__ float factors[];
    float maximum=-__int_as_float(0x7f800000);
    for (U i=first+threadIdx.x;i<count;i+=256) maximum=fmaxf(maximum,partial[(h*capacity+i)*(head+2)+head]);
    maximum=max_block(maximum);
    float denom=0;
    for (U i=first+threadIdx.x;i<count;i+=256) {
        U base=(h*capacity+i)*(head+2);
        float factor=expf(partial[base+head]-maximum);
        factors[i]=factor;denom+=partial[base+head+1]*factor;
    }
    denom=sum_block(denom);
    if (threadIdx.x<head) {
        float acc=0;
        for (U i=first;i<count;++i) acc=__fmaf_rn(partial[(h*capacity+i)*(head+2)+threadIdx.x],factors[i],acc);
        out[h*head+threadIdx.x]=acc/denom;
    }
}
