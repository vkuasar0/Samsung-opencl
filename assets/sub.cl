__kernel void sub(__global uchar3 *image1, __global uchar3 *image2,
                  __global uchar3 *output, int width, int height) {
  int2 gid = (int2)(get_global_id(0), get_global_id(1));
  int idx = gid.y * width + gid.x;
  if (gid.x < width && gid.y < height) {
    uchar4 val1 = vload4(0, (__global uchar *)(image1 + idx));
    uchar4 val2 = vload4(0, (__global uchar *)(image2 + idx));
    float4 diff = convert_float4(val1) - convert_float4(val2);
    diff = clamp(diff, 0.0f, 255.0f);
    uchar4 out = convert_uchar4(diff);
    vstore4(out, 0, (__global uchar *)(output + idx));
  }
}
