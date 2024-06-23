__kernel void div(__global uchar3 *image1, float divisor,
                  __global uchar3 *output, int width, int height) {
  int2 gid = (int2)(get_global_id(0), get_global_id(1));
  int idx = gid.y * width + gid.x;
  if (gid.x < width && gid.y < height) {
    uchar4 val1 = vload4(0, (__global uchar *)(image1 + idx));
    float4 div = convert_float4(val1) / divisor;
    div = round(div);
    div = clamp(div, 0.0f, 255.0f);
    uchar4 out = convert_uchar4(div);
    vstore4(out, 0, (__global uchar *)(output + idx));
  }
}


