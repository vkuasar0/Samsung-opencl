__kernel void gaussian(__global const uchar *inputImage,
                       __global uchar *outputImage, const int width,
                       const int height, const int kernelSize,
                       const float sigma) {
  const int channels = 3;
  const int rowSize = width * channels;
  const int radius = kernelSize / 2;

  int x = get_global_id(0);
  int y = get_global_id(1);

  if (x < width && y < height) {
    float sum_r = 0.0f;
    float sum_g = 0.0f;
    float sum_b = 0.0f;
    float weight_sum = 0.0f;

    for (int i = -radius; i <= radius; i++) {
      for (int j = -radius; j <= radius; j++) {
        int px = x + j;
        int py = y + i;

        if (px >= 0 && px < width && py >= 0 && py < height) {
          int index = (py * rowSize) + (px * channels);

          float weight = exp(-(i * i + j * j) / (2.0f * sigma * sigma));
          weight_sum += weight;

          sum_r += inputImage[index] * weight;
          sum_g += inputImage[index + 1] * weight;
          sum_b += inputImage[index + 2] * weight;
        }
      }
    }

    outputImage[(y * rowSize) + (x * channels)] = (uchar)(sum_r / weight_sum);
    outputImage[(y * rowSize) + (x * channels) + 1] =
        (uchar)(sum_g / weight_sum);
    outputImage[(y * rowSize) + (x * channels) + 2] =
        (uchar)(sum_b / weight_sum);
  }
}
