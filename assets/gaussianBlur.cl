__kernel void gaussian(__global const uchar4 *inputImage,
                       __global uchar4 *outputImage, const int width,
                       const int height, const int kernelSize,
                       const float sigma) {
  const int radius = kernelSize / 2;

  int x = get_global_id(0);
  int y = get_global_id(1);

  if (x < width && y < height) {
    float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float weight_sum = 0.0f;

    for (int i = -radius; i <= radius; i++) {
      for (int j = -radius; j <= radius; j++) {
        int px = x + j;
        int py = y + i;

        if (px >= 0 && px < width && py >= 0 && py < height) {
          int index = (py * width) + px;

          float weight = exp(-(i * i + j * j) / (2.0f * sigma * sigma));
          weight_sum += weight;

          sum += (float4)(inputImage[index].x, inputImage[index].y, inputImage[index].z, inputImage[index].w) * weight;
        }
      }
    }

    sum /= weight_sum;
    outputImage[y * width + x] = (uchar4)(
        (uchar)sum.x,
        (uchar)sum.y,
        (uchar)sum.z,
        (uchar)sum.w
    );
  }
}
