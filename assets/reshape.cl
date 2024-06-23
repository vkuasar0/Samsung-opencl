__kernel void reshape(__global uchar *src, __global uchar *dst,
                            const int src_rows, const int src_cols,
                            const int dst_rows, const int dst_cols,
                            const int src_channels, const int dst_channels) {
  int gid_x = get_global_id(0);
  int gid_y = get_global_id(1);
  int gid_c = get_global_id(2);

  if (gid_x >= dst_cols || gid_y >= dst_rows)
    return;

  float src_yf = (gid_y + 0.5f) * src_rows / dst_rows - 0.5f;
  int src_y1 = (int)floor(src_yf);
  int src_y2 = (int)ceil(src_yf);

  float y_weight2 = src_yf - src_y1;
  float y_weight1 = 1.0f - y_weight2;

  int src_x = (gid_x * src_cols) / dst_cols;

  int src_index1 =
      (src_y1 * src_cols + gid_x % src_cols) * src_channels + gid_c;
  int src_index2 =
      (src_y2 * src_cols + gid_x % src_cols) * src_channels + gid_c;

  int dst_index = (gid_y * dst_cols + gid_x) * dst_channels;

  for (int c = 0; c < dst_channels; ++c) {
    float val1 = src[src_index1 + c];
    float val2 = src[src_index2 + c];
    dst[dst_index + c] = (uchar)(val1 * y_weight1 + val2 * y_weight2);
  }
}
