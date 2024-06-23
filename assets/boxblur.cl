/*__kernel void box(__global const uchar *inputImage, __global uchar *outputImage,
                  const int width, const int height, const int kernelSize) {
  const int channels = 3;
  const int rowSize = width * channels;
  const int radius = kernelSize / 2;
  int x = get_global_id(0);
  int y = get_global_id(1);

  float4 sum = (float4)(0.0f);
  int count = 0;

  // Vectorize the loop by processing eight neighboring pixels at once
  for (int i = -radius; i <= radius; i += 2) {
    for (int j = -radius; j <= radius; j += 8) {
      // Load eight neighboring pixels at once as an uchar4 vector
      int px = x + j;
      int py = y + i;
      uchar4 pixels_uchar =
          vload4((py * rowSize) + (px * channels), inputImage);

      // Convert uchar4 to float4
      float4 pixels = convert_float4(pixels_uchar);

      // Accumulate the color values and count of neighboring pixels
      sum = mad(pixels, (float4)(1.0f), sum);
      count += 8;
    }
  }

  // Compute the average color value
  uchar3 outputPixel;
  outputPixel.x = (uchar)(sum.x / count + 0.5f);
  outputPixel.y = (uchar)(sum.y / count + 0.5f);
  outputPixel.z = (uchar)(sum.z / count + 0.5f);

  // Store the average color value to the output image
  vstore3(outputPixel, (y * rowSize) + (x * channels), outputImage);
}*/


/*__kernel void box(__global const uchar *inputImage, __global uchar *outputImage,
                  const int width, const int height, const int kernelSize) {
  const int channels = 3;
  const int rowSize = width * channels;
  const int radius = kernelSize / 2;
  int x = get_global_id(0);
  int y = get_global_id(1);

  if (x < width && y < height) {
    int sum_r = 0;
    int sum_g = 0;
    int sum_b = 0;
    int count = 0;
    int px, py, tempvar;
#pragma unroll
    for (int i = -radius; i <= radius; i += 2) {
      py = y + i;
      for (int j = -radius; j <= radius; j += 2) {
        px = x + j;

        // Load two neighboring pixels at once
        if (px >= 0 && px < width && py >= 0 && py < height) {
          int index1 = (py * rowSize) + (px * channels);
          uchar3 pixel1 = vload3(index1, inputImage);
          sum_r += pixel1.x;
          sum_g += pixel1.y;
          sum_b += pixel1.z;
          count++;
        }
        if (j + 1 <= radius) {
          int px2 = x + j + 1;
          if (px2 >= 0 && px2 < width && py >= 0 && py < height) {
            int index2 = (py * rowSize) + (px2 * channels);
            uchar3 pixel2 = vload3(index2, inputImage);
            sum_r += pixel2.x;
            sum_g += pixel2.y;
            sum_b += pixel2.z;
            count++;
          }
        }
      }
    }

    uchar3 outputPixel;
    outputPixel.x = (uchar)((float)sum_r / count + 0.5f);
    outputPixel.y = (uchar)((float)sum_g / count + 0.5f);
    outputPixel.z = (uchar)((float)sum_b / count + 0.5f);
    vstore3(outputPixel, (y * rowSize) + (x * channels), outputImage);
  }
}
*/
/*__kernel void box(__global const uchar* inputImage, __global uchar* outputImage, const int width, const int height,const int kernelSize) {
    const int channels = 3;
    const int rowSize = width * channels;
    const int radius = kernelSize/2;
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width && y < height) {
        int sum_r = 0;
        int sum_g = 0;
        int sum_b = 0;
        int count = 0;
        for (int i = -radius; i <=radius; i++) {
            for (int j = -radius; j <=radius; j++) {
                int px = x + j;
                int py = y + i;
                if (px >= 0 && px < width && py >= 0 && py < height) {
                    int index = (py * rowSize) + (px * channels);
                    sum_r += inputImage[index];
                    sum_g += inputImage[index+1];
                    sum_b += inputImage[index+2];
                    count++;
                }
            }
        }
        outputImage[(y * rowSize) + (x * channels)] = (uchar)(sum_r / count);
        outputImage[(y * rowSize) + (x * channels) + 1] = (uchar)(sum_g / count);
        outputImage[(y * rowSize) + (x * channels) + 2] = (uchar)(sum_b / count);
    }
}
*/
// __kernel void box(__global const uchar* inputImage, __global uchar* outputImage, const int width, const int height, const int kernelSize) {
//     const int channels = 3;
//     const int rowSize = width * channels;
//     const int radius = kernelSize / 2;
//     int x = get_global_id(0);
//     int y = get_global_id(1);

//     if (x < width && y < height) {
//         float sum_r = 0.0f;
//         float sum_g = 0.0f;
//         float sum_b = 0.0f;
//         int count = 0;
//         for (int i = -radius; i <= radius; i++) {
//             for (int j = -radius; j <= radius; j++) {

//                 int px = x + j;
//                 int py = y + i;
//                 if (px < 0)
//                     px = -px;
//                 else if (px >= width)
//                     px = (2 * width) - px - 2;
//                 if (py < 0)
//                     py = -py;
//                 else if (py >= height)
//                     py = (2 * height) - py - 2;

//                 int index = (py * rowSize) + (px * channels);
//                 sum_r += (float)inputImage[index];
//                 sum_g += (float)inputImage[index + 1];
//                 sum_b += (float)inputImage[index + 2];
//                 count++;
//             }
//         }
//         outputImage[(y * rowSize) + (x * channels)] = (uchar)round(sum_r / count);
//         outputImage[(y * rowSize) + (x * channels) + 1] = (uchar)round(sum_g / count);
//         outputImage[(y * rowSize) + (x * channels) + 2] = (uchar)round(sum_b / count);
//     }
// }

#define channels 3
__kernel void box(__global const uchar* inputImage, __global uchar* outputImage, const int width, const int height, const int kernelSize,const int count) {
    
    const int rowSize = width * channels;
    const int radius = kernelSize / 2;
    int2 gid = (int2)(get_global_id(0), get_global_id(1));
    // int chan = get_local_id(0);
        float3 sum = (float3)(0.0f);
       

        for (int i = -radius; i <= radius; i++) {
            for (int j = -radius; j <= radius; j++) {
                int px = gid.x + j;
                int py = gid.y + i;
                px = px<0?-px:(px>=width)?(2*width)-px-2:px;
                py = py<0?-py:(py>=height)?(2*height)-py-2:py;

                int index = (py * rowSize) + (px * channels);
                uchar3 pixel = vload3(0, (__global uchar*)(inputImage +index));
                //   uchar pixel =  (__global uchar*)(inputImage +index);
    
                // sum += convert_float3(pixel);
                sum += convert_float3(pixel);
    
        
            }
        }


        float3 avg = sum / count;
        uchar3 result = convert_uchar3(round(avg));
        vstore3(result, 0, (__global uchar*)(outputImage + (gid.y * rowSize) + (gid.x * channels)));
}

// #define channels 3
// __kernel void box(__global const uchar* inputImage, __global uchar* outputImage, const int width, const int height, const int kernelSize) {
    
//     const int rowSize = width * channels;
//     const int radius = kernelSize / 2;
//     int2 gid = (int2)(get_global_id(0), get_global_id(1));

//     if (gid.x < width && gid.y < height) {
//         float3 sum = (float3)(0.0f, 0.0f, 0.0f);
//         int count = 0;

//         for (int i = -radius; i <= radius; i++) {
//             for (int j = -radius; j <= radius; j++) {
//                 int px = gid.x + j;
//                 int py = gid.y + i;
//                 if (px < 0)
//                     px = -px;
//                 else if (px >= width)
//                     px = (2 * width) - px - 2;
//                 if (py < 0)
//                     py = -py;
//                 else if (py >= height)
//                     py = (2 * height) - py - 2;

//                 int index = (py * rowSize) + (px * channels);
//                 uchar3 pixel = vload3(0, (__global uchar*)(inputImage +index));
//                 sum.x += convert_float(pixel.x);
//                 sum.y += convert_float(pixel.y);
//                 sum.z += convert_float(pixel.z);
//                 count++;
//             }
//         }

//         float3 avg = sum / count;
//         uchar3 result = convert_uchar3(round(avg));
//         vstore3(result, 0, (__global uchar*)(outputImage + (gid.y * rowSize) + (gid.x * channels)));
//     }
// }

/*__kernel void box(__global const uchar* inputImage, __global uchar*
outputImage, const int width, const int height,const int kernelSize) { const int
channels = 3; const int rowSize = width * channels; const int radius =
kernelSize/2; int x = get_global_id(0); int y = get_global_id(1);

    if (x < width && y < height) {
        int sum_r = 0;
        int sum_g = 0;
        int sum_b = 0;
        int count = 0;

        // Loop unrolling
        for (int i = -radius; i <= radius; i += 2) {
            for (int j = -radius; j <= radius; j += 2) {
                int px = x + j;
                int py = y + i;

                // Load two neighboring pixels at once
                if (px >= 0 && px < width && py >= 0 && py < height) {
                    int index1 = (py * rowSize) + (px * channels);
                    uchar3 pixel1 = vload3(index1, inputImage);
                    sum_r += pixel1.x;
                    sum_g += pixel1.y;
                    sum_b += pixel1.z;
                    count++;
                }
                if (j + 1 <= radius) {
                    int px2 = x + j + 1;
                    if (px2 >= 0 && px2 < width && py >= 0 && py < height) {
                        int index2 = (py * rowSize) + (px2 * channels);
                        uchar3 pixel2 = vload3(index2, inputImage);
                        sum_r += pixel2.x;
                        sum_g += pixel2.y;
                        sum_b += pixel2.z;
                        count++;
                    }
                }
            }
        }

        uchar3 outputPixel;
        outputPixel.x = (uchar)((float)sum_r / count + 0.5f);
        outputPixel.y = (uchar)((float)sum_g / count + 0.5f);
        outputPixel.z = (uchar)((float)sum_b / count + 0.5f);
        vstore3(outputPixel, (y * rowSize) + (x * channels), outputImage);
    }
}
*/
