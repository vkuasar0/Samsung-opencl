__kernel void merge(__global uchar *R, __global uchar *G, __global uchar *B,
                    __global uchar *output, const int width, const int height) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if (x >= width || y >= height)
    return;

  const int index = (y * width + x) * 3;

  output[index] = R[y * width + x];
  output[index + 1] = G[y * width + x];
  output[index + 2] = B[y * width + x];
}
