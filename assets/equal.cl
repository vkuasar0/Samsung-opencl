__kernel void equal(__global uchar3 *image1, __global uchar3 *image2,
                    __global uchar3 *dst, int width, int height) {
  int2 gid = (int2)(get_global_id(0), get_global_id(1));
  int idx = gid.y * width + gid.x;
  if (gid.x < width && gid.y < height) {
    uchar4 val1 = vload4(0, (__global uchar *)(image1 + idx));
    uchar4 val2 = vload4(0, (__global uchar *)(image2 + idx));
    int4 cmp = convert_int4(val1 == val2);
    uchar4 out = convert_uchar4(cmp);
    vstore4(out, 0, (__global uchar *)(dst + idx));
  }
}
