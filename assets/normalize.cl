__kernel void normalized(__global uchar *input, __global uchar *output,
                         const int width, const int height, const float minVal,
                         const float maxVal) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  if (x < width && y < height) {
    int in_idx = y * width * 3 + x * 3;
    int out_idx = y * width * 3 + x * 3;

    for (int i = 0; i < 3; i++) {
      float val = (float)input[in_idx + i];
      float scaled_val = (val - minVal) * (maxVal - 0.0f) / (255.0f - 0.0f) +
                         0.5f; // Scale the value to the specified range and
                               // round to the nearest integer
      output[out_idx + i] =
          (uchar)(scaled_val > 255.0f
                      ? 255.0f
                      : scaled_val); // Clamp the value to 255 if it exceeds the
                                     // maximum value
    }
  }
}
