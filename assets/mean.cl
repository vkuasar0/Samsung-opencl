__kernel void mean(__global uchar3* src1, __global uchar3* src2,__global uchar3* dst,
float alpha,float beta,float gamma, int width,int height)
{
    int2 gid = (int2)(get_global_id(0), get_global_id(1));
    int idx = gid.y * width + gid.x;

    if (gid.x < width && gid.y < height)
    {
        uchar4 val1 = vload4(0, (__global uchar *)(src1 + idx));
        uchar4 val2 = vload4(0, (__global uchar *)(src2 + idx));

        float4 fval1 = convert_float4(val1);
        float4 fval2 = convert_float4(val2);

        float4 res = alpha * fval1 + beta * fval2 + gamma;

        uchar4 out = convert_uchar4(clamp(res, 0.0f, 255.0f));
        vstore4(out, 0, (__global uchar *)(dst + idx));
    }
}


