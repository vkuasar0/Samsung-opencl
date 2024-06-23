__kernel void rgb(__global uchar *src_img, __global uchar *dst_img,
                  const int cols, const int rows) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if (x >= cols || y >= rows) {
    return;
  }
  int src_index = (y * cols + x) * 3;
  int dst_index = src_index;

  float Y = (float)src_img[src_index++];
  float Cr = (float)src_img[src_index++] - 128.0f;
  float Cb = (float)src_img[src_index] - 128.0f;

  int R = round(Y + 1.403f * Cr);
  int G = round(Y - 0.344f * Cb - 0.714f * Cr);
  float B = round(Y + 1.773f * Cb);

  dst_img[dst_index++] = clamp((int)R, 0, 255);
  dst_img[dst_index++] = clamp((int)G, 0, 255);
  dst_img[dst_index] = clamp((int)B, 0, 255);
}

// Kernel execution time: 1.5209 ms
