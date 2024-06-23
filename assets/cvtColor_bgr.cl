__kernel void rgb(__global const uchar *inputImage, __global uchar *outputImage,
                  int width, int height) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  if (x < width && y < height) {
    int index = (y * width + x) * 3;
    uchar r = inputImage[index];
    uchar g = inputImage[index + 1];
    uchar b = inputImage[index + 2];

    outputImage[index] = b;
    outputImage[index + 1] = g;
    outputImage[index + 2] = r;
  }
}
