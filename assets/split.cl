__kernel void split(__global uchar *input, __global uchar *output_r,
                    __global uchar *output_g, __global uchar *output_b,
                    const int width, const int height) {
  const int gid_x = get_global_id(0);
  const int gid_y = get_global_id(1);

  if (gid_x >= width || gid_y >= height)
    return;

  const int input_idx = (gid_y * width + gid_x) * 3;
  const int output_idx = gid_y * width + gid_x;

  output_r[output_idx] = input[input_idx];
  output_g[output_idx] = input[input_idx + 1];
  output_b[output_idx] = input[input_idx + 2];
}
