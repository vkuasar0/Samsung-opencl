__kernel void bgr(__global const uchar *grayImage, __global uchar *bgrImage,
                  int width, int height) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  if (x < width && y < height) {
    int grayIndex = y * width + x;
    int bgrIndex = grayIndex * 3;

    uchar gray = grayImage[grayIndex];

    bgrImage[bgrIndex++] = gray;
    bgrImage[bgrIndex++] = gray;
    bgrImage[bgrIndex] = gray;
  }
}
// Kernel execution time: 0.997888 ms
