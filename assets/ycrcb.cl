/*__kernel void rgb2(__global const uchar *inputImage,
                   __global uchar *outputImage, const int width,
                   const int height) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if (x >= width || y >= height) {
    return;
  }

  const int idx = y * width + x;
  const uchar R = inputImage[3 * idx];
  const uchar G = inputImage[3 * idx + 1];
  const uchar B = inputImage[3 * idx + 2];

 const float Y = round(0.299f * R + 0.587f * G + 0.114f * B);
  const float Cr = round(128 + 0.5f * R - 0.4187f * G - 0.0813f * B);
  const float Cb = round(128 - 0.1687f * R - 0.3313f * G + 0.5f * B);
  
  outputImage[3 * idx] = (uchar)(clamp((int)Y,0,255));
  outputImage[3 * idx + 1] = (uchar)(clamp((int)Cr,0,255));
  outputImage[3 * idx + 2] = (uchar)(clamp((int)Cb,0,255));
}
// Kernel execution time: 1.51398 ms
// Total Execution Time : 180.680000 m*/

__kernel void rgb2(__global const uchar *inputImage,
                   __global uchar *outputImage, const int width,
                   const int height) {
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if (x >= width || y >= height) {
    return;
  }

  const int idx = y * width + x;
  const uchar R = inputImage[3 * idx];
  const uchar G = inputImage[3 * idx + 1];
  const uchar B = inputImage[3 * idx + 2];

 const float Y = round(0.299f * R + 0.587f * G + 0.114f * B);
  const float Cr = round((R-Y)*0.713f+128);
  const float Cb = round((B-Y)*0.564f+128);
  
  outputImage[3 * idx] = (uchar)(clamp((int)Y,0,255));
  outputImage[3 * idx + 1] = (uchar)(clamp((int)Cr,0,255));
  outputImage[3 * idx + 2] = (uchar)(clamp((int)Cb,0,255));
}
// Kernel execution time: 1.51398 ms
// Total Execution Time : 180.680000 m
