__kernel void mul(__global uchar3 *input, float multiplier,
                  __global uchar3 *output, int width, int height) {
  int2 gid = (int2)(get_global_id(0), get_global_id(1));
  int idx = gid.y * width + gid.x;
  if (gid.x < width && gid.y < height) {
    uchar4 val = vload4(0, (__global uchar *)(input + idx));
    float4 product = convert_float4(val) * multiplier;
    product = clamp(product, 0.0f, 255.0f);
    uchar4 out = convert_uchar4(product);
    vstore4(out, 0, (__global uchar *)(output + idx));
  }
}
