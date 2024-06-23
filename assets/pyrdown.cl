// Using Gaussian pyramid

#define size 5
__kernel void pyrdown(__global uchar *src, __global uchar *dst, int out_width,
                      int out_height, int in_width, int in_height) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  if (x < out_width && y < out_height) {

    int dst_index = y * out_width * 3 + x * 3;

    float kenel[size][size] = {
      {0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625},
      {0.015625, 0.0625, 0.09375, 0.0625, 0.015625},
      {0.0234375, 0.09375, 0.140625, 0.09375, 0.0234375},
      {0.015625, 0.0625, 0.09375, 0.0625, 0.015625},
      {0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625}};

    float r = 0.0f, g = 0.0f, b = 0.0f;
    float sum = 0.0f;
    int px, py;

    #pragma unroll
    for (int ky = -2; ky <= 2; ky++) {
      py = y * 2 + ky;
      for (int kx = -2; kx <= 2; kx++) {
        px = x * 2 + kx;

        // Apply border reflection for out-of-bound indices
        if (px < 0)
          px = -px;
        else if (px >= in_width)
          px = (2 * in_width) - px - 2;
        if (py < 0)
          py = -py;
        else if (py >= in_height)
          py = (2 * in_height) - py - 2;

        float w = kenel[ky + 2][kx + 2];
        int index = py * in_width * 3 + px * 3;

        r += (w * (float)src[index++]);
        g += (w * (float)src[index++]);
        b += (w * (float)src[index]);
      }
    }
    dst[dst_index++] = (uchar)clamp((int)round(r), 0, 255);
    dst[dst_index++] = (uchar)clamp((int)round(g), 0, 255);
    dst[dst_index] = (uchar)clamp((int)round(b), 0, 255);
  }
}
// #define size 5
// __kernel void pyrdown(__global uchar *src, __global uchar *dst, int out_width,
//                       int out_height, int in_width, int in_height) {
//   int x = get_global_id(0);
//   int y = get_global_id(1);

//   if (x < out_width && y < out_height) {

//     int dst_index = y * out_width * 3 + x * 3;

//     float kenel[size][size] = {
//       {0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625},
//       {0.015625, 0.0625, 0.09375, 0.0625, 0.015625},
//       {0.0234375, 0.09375, 0.140625, 0.09375, 0.0234375},
//       {0.015625, 0.0625, 0.09375, 0.0625, 0.015625},
//       {0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625}};

//     float r = 0.0f, g = 0.0f, b = 0.0f;
//     float sum = 0.0f;
//     int px, py;

//     #pragma unroll
//     for (int ky = -2; ky <= 2; ky++) {
//       py = y * 2 + ky;
//       for (int kx = -2; kx <= 2; kx++) {
//         px = x * 2 + kx;
//         // Apply border reflection for out-of-bound indices
//         if (px < 0)
//           px = -px;
//         else if (px >= in_width)
//           px = (2 * in_width) - px - 2;
//         if (py < 0)
//           py = -py;
//         else if (py >= in_height)
//           py = (2 * in_height) - py - 2;

//         float w = kenel[ky + 2][kx + 2];
//         int index = py * in_width * 3 + px * 3;

//         r += (w * (float)src[index++]);
//         g += (w * (float)src[index++]);
//         b += (w * (float)src[index]);

//         sum += w;
//       }
//     }
//     r /= sum;
//     g /= sum;
//     b /= sum;
//     dst[dst_index++] = (uchar)clamp((int)round(r), 0, 255);
//     dst[dst_index++] = (uchar)clamp((int)round(g), 0, 255);
//     dst[dst_index] = (uchar)clamp((int)round(b), 0, 255);
//   }
// }

// #define size 5
// __kernel void pyrdown(__global uchar *src, __global uchar *dst, int width,
//                       int height) {
//   int x = get_global_id(0);
//   int y = get_global_id(1);

//   if (x < width && y < height && x%2==0 && y%2==0) {

//     int dst_index = ((int)round(((float)y / 2))) * (width / 2) *3+
//                     ((int)round(((float)x / 2))) *3;

//       if(dst_index==31752)
//         printf("OCL: x: %d, y: %d, dst_index: %d\n", x, y, dst_index);


// float kenel[size][size] = {
//       {0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625},
//       {0.015625, 0.0625, 0.09375, 0.0625, 0.015625},
//       {0.0234375, 0.09375, 0.140625, 0.09375, 0.0234375},
//       {0.015625, 0.0625, 0.09375, 0.0625, 0.015625},
//       {0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625}};

//     float r = 0.0f, g = 0.0f, b = 0.0f;
//     // float sum = 0.0f;
//     int px, py;

// #pragma unroll
//     for (int ky = -2; ky <= 2; ky++) {
//       py = y + ky;

//       if (py < 0)
//           py = -py;
//       else if (py >= height)
//         py = (2 * height) - py - 2;

//       for (int kx = -2; kx <= 2; kx++) {
//         px = x + kx;
        
//         // if(px<0 || px>=width || py<0 || py>=height)
//         //   continue;
//         // Apply border reflection for out-of-bound indices
//         if (px < 0)
//           px = -px;
//         else if (px >= width)
//           px = (2 * width) - px - 2;

//         float w = kenel[ky + 2][kx + 2];
//         // w = w / 256.f;

//         int index = py * width *3 + px *3;

//         r += (w * (float)src[index++]);
//         g += (w * (float)src[index++]);
//         b += (w * (float)src[index]);
//       }
//     }

//     barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

//     if(dst_index==31752)
//           printf("OCL: (x, y): (%d, %d), r: %f, g: %f, b: %f, index: %d\n", x, y, r, g, b, dst_index);
//     dst[dst_index++] = (uchar)clamp((int)round(r), 0, 255);
//     dst[dst_index++] = (uchar)clamp((int)round(g), 0, 255);
//     dst[dst_index] = (uchar)clamp((int)round(b), 0, 255);
//   }
// }

// #define size 5
// __kernel void pyrdown(__global uchar *src, __global uchar *dst, int width,
//                       int height) {
//   int x = get_global_id(0);
//   int y = get_global_id(1);

//   if (x < width && y < height) {

//     int dst_index = ((int)round(((float)y / 2))) * (width / 2)+
//                     ((int)round(((float)x / 2)));

//     // float kenel[size][size] = {
//     //     {1.f, 4.f, 6.f, 4.f, 1.f},
//     //     {4.f, 16.f, 24.f, 16.f, 4.f},
//     //     {6.f, 24.f, 36.f, 24.f, 6.f},
//     //     {4.f, 16.f, 24.f, 16.f, 4.f},
//     //     {1.f, 4.f, 6.f, 4.f, 1.f}
//     // };
// float kenel[size][size] = {
//       {0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625},
//       {0.015625, 0.0625, 0.09375, 0.0625, 0.015625},
//       {0.0234375, 0.09375, 0.140625, 0.09375, 0.0234375},
//       {0.015625, 0.0625, 0.09375, 0.0625, 0.015625},
//       {0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625}};

//     float r = 0.0f, g = 0.0f, b = 0.0f;
//     float sum = 0.0f;
//     int px, py;

// #pragma unroll
//     for (int ky = -2; ky <= 2; ky++) {
//       py = y + ky;
//       for (int kx = -2; kx <= 2; kx++) {
//         px = x + kx;
        
//         // if(px<0 || px>=width || py<0 || py>=height)
//         //   continue;
//         // Apply border reflection for out-of-bound indices
//         if (px < 0)
//           px = -px;
//         else if (px >= width)
//           px = (2 * width) - px - 2;
//         if (py < 0)
//           py = -py;
//         else if (py >= height)
//           py = (2 * height) - py - 2;

//         float w = kenel[ky + 2][kx + 2];
//         // w = w / 256.f;

//         int index = py * width + px;

//         r += (w * (float)src[index++]);
//         // g += (w * (float)src[index++]);
//         // b += (w * (float)src[index]);

//         sum += w;
//       }
//     }
//     r /= sum;
//     // g /= sum;
//     // b /= sum;
//     dst[dst_index] = (uchar)clamp((int)round(r), 0, 255);
//     // dst[dst_index++] = (uchar)clamp((int)round(g), 0, 255);
//     // dst[dst_index] = (uchar)clamp((int)round(b), 0, 255);
//   }
// }

// #define size 5
// __kernel void pyrdown(__global uchar *src, __global uchar *dst, int width,
//                       int height) {
//   int x = get_global_id(0);
//   int y = get_global_id(1);

//   if (x < width && y < height && x%2==0 && y%2==0) {

//     int dst_index = ((int)round(((float)y / 2))) * (width / 2) *3+
//                     ((int)round(((float)x / 2))) *3;

//       if(dst_index==31752)
//         printf("OCL: x: %d, y: %d, dst_index: %d\n", x, y, dst_index);


// float kenel[size][size] = {
//       {0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625},
//       {0.015625, 0.0625, 0.09375, 0.0625, 0.015625},
//       {0.0234375, 0.09375, 0.140625, 0.09375, 0.0234375},
//       {0.015625, 0.0625, 0.09375, 0.0625, 0.015625},
//       {0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625}};

//     float r = 0.0f, g = 0.0f, b = 0.0f;
//     // float sum = 0.0f;
//     int px, py;

// #pragma unroll
//     for (int ky = -2; ky <= 2; ky++) {
//       py = y + ky;

//       if (py < 0)
//           py = -py;
//       else if (py >= height)
//         py = (2 * height) - py - 2;

//       for (int kx = -2; kx <= 2; kx++) {
//         px = x + kx;
        
//         // if(px<0 || px>=width || py<0 || py>=height)
//         //   continue;
//         // Apply border reflection for out-of-bound indices
//         if (px < 0)
//           px = -px;
//         else if (px >= width)
//           px = (2 * width) - px - 2;

//         float w = kenel[ky + 2][kx + 2];
//         // w = w / 256.f;

//         int index = py * width *3 + px *3;

//         r += (w * (float)src[index++]);
//         g += (w * (float)src[index++]);
//         b += (w * (float)src[index]);
//       }
//     }

//     barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

//     if(dst_index==31752)
//           printf("OCL: (x, y): (%d, %d), r: %f, g: %f, b: %f, index: %d\n", x, y, r, g, b, dst_index);
//     dst[dst_index++] = (uchar)clamp((int)round(r), 0, 255);
//     dst[dst_index++] = (uchar)clamp((int)round(g), 0, 255);
//     dst[dst_index] = (uchar)clamp((int)round(b), 0, 255);
//   }
// }

// #define size 5
// __kernel void pyrdown(__global uchar *src, __global uchar *dst, int width,
//                       int height) {
//   int x = get_global_id(0);
//   int y = get_global_id(1);

//   if (x < width && y < height) {

//     int dst_index = ((int)round(((float)y / 2))) * (width / 2)+
//                     ((int)round(((float)x / 2)));

//     // float kenel[size][size] = {
//     //     {1.f, 4.f, 6.f, 4.f, 1.f},
//     //     {4.f, 16.f, 24.f, 16.f, 4.f},
//     //     {6.f, 24.f, 36.f, 24.f, 6.f},
//     //     {4.f, 16.f, 24.f, 16.f, 4.f},
//     //     {1.f, 4.f, 6.f, 4.f, 1.f}
//     // };
// float kenel[size][size] = {
//       {0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625},
//       {0.015625, 0.0625, 0.09375, 0.0625, 0.015625},
//       {0.0234375, 0.09375, 0.140625, 0.09375, 0.0234375},
//       {0.015625, 0.0625, 0.09375, 0.0625, 0.015625},
//       {0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625}};

//     float r = 0.0f, g = 0.0f, b = 0.0f;
//     float sum = 0.0f;
//     int px, py;

// #pragma unroll
//     for (int ky = -2; ky <= 2; ky++) {
//       py = y + ky;
//       for (int kx = -2; kx <= 2; kx++) {
//         px = x + kx;
        
//         // if(px<0 || px>=width || py<0 || py>=height)
//         //   continue;
//         // Apply border reflection for out-of-bound indices
//         if (px < 0)
//           px = -px;
//         else if (px >= width)
//           px = (2 * width) - px - 2;
//         if (py < 0)
//           py = -py;
//         else if (py >= height)
//           py = (2 * height) - py - 2;

//         float w = kenel[ky + 2][kx + 2];
//         // w = w / 256.f;

//         int index = py * width + px;

//         r += (w * (float)src[index++]);
//         // g += (w * (float)src[index++]);
//         // b += (w * (float)src[index]);

//         sum += w;
//       }
//     }
//     r /= sum;
//     // g /= sum;
//     // b /= sum;
//     dst[dst_index] = (uchar)clamp((int)round(r), 0, 255);
//     // dst[dst_index++] = (uchar)clamp((int)round(g), 0, 255);
//     // dst[dst_index] = (uchar)clamp((int)round(b), 0, 255);
//   }
// }

// #define size 5
// __kernel void pyrdown(__global uchar* src, __global uchar* dst, int width, int height)
// {
//     int x = get_global_id(0);
//     int y = get_global_id(1);

//     if (x < width && y < height){
//         // int src_index = y * width * 3 + x * 3;
//         int dst_index = (y / 2) * (width / 2) * 3 + (x / 2) * 3;

//         // float kenel[size][size] = { { 0.003765, 0.015019, 0.023792, 0.015019, 0.003765 },
//         //                     { 0.015019, 0.059912, 0.094907, 0.059912, 0.015019 },
//         //                     { 0.023792, 0.094907, 0.150342, 0.094907, 0.023792 },
//         //                     { 0.015019, 0.059912, 0.094907, 0.059912, 0.015019 },
//         //                     { 0.003765, 0.015019, 0.023792, 0.015019, 0.003765 } };

//         float kenel[size][size] = { { 0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625 },
//                             { 0.015625, 0.0625, 0.09375, 0.0625, 0.015625 },
//                             { 0.0234375, 0.09375, 0.140625, 0.09375, 0.0234375 },
//                             { 0.015625, 0.0625, 0.09375, 0.0625, 0.015625 },
//                             { 0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625 } };

//         float r = 0.0f, g = 0.0f, b = 0.0f;
//         float sum = 0.0f;
//         int px, py;

//         // #pragma unroll
//         for (int ky = -2; ky <= 2; ky++)
//         {
//             py = y + ky;
//             if(py >=0 && py < height){
//                 for (int kx = -2; kx <= 2; kx++)
//                 {
//                     px = x + kx;

//                     if (px < 0 || px >= width)
//                         continue;

//                     float w = kenel[ky+2][kx+2];

//                     int index = py * width * 3 + px * 3;

//                     r += w * (float)src[index++];
//                     g += w * (float)src[index++];
//                     b += w * (float)src[index];

//                     sum += w;
//                 }
//             }
//         }
//         r /= sum;
//         g /= sum;
//         b /= sum;
//         dst[dst_index++] = (uchar)(r > 255.0f ? 255.0f : r);
//         dst[dst_index++] = (uchar)(g > 255.0f ? 255.0f : g);
//         dst[dst_index] = (uchar)(b > 255.0f ? 255.0f : b);
//     }
// }

// #define BORDER_REFLECT 1

// __kernel void pyrdown(__global uchar *src, __global uchar *dst, int width,
//                       int height) {
//   int x = get_global_id(0);
//   int y = get_global_id(1);

//   if (x < width && y < height) {
//     int dst_index = ((int)round(((float)y / 2))) * (width / 2) * 3 +
//                     ((int)round(((float)x / 2))) * 3;

//     float kenel[size][size] = {
//         {0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625},
//         {0.015625, 0.0625, 0.09375, 0.0625, 0.015625},
//         {0.0234375, 0.09375, 0.140625, 0.09375, 0.0234375},
//         {0.015625, 0.0625, 0.09375, 0.0625, 0.015625},
//         {0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625}
//     };

//     float r = 0.0f, g = 0.0f, b = 0.0f;
//     float sum = 0.0f;
//     int px, py;

// #pragma unroll
//     for (int ky = -2; ky <= 2; ky++) {
//       py = y + ky;

// #if BORDER_REFLECT
//       if (py < 0)
//           py = -py;
//       else if (py >= height)
//           py = (2 * height) - py - 2;
// #else
//       if (py < 0 || py >= height)
//           continue;
// #endif

//       for (int kx = -2; kx <= 2; kx++) {
//         px = x + kx;

// #if BORDER_REFLECT
//         if (px < 0)
//           px = -px;
//         else if (px >= width)
//           px = (2 * width) - px - 2;
// #else
//         if (px < 0 || px >= width)
//           continue;
// #endif

//         float w = kenel[ky + 2][kx + 2];
//         w /= 256;

//         int index = py * width * 3 + px * 3;

//         r += w * (float)src[index++];
//         g += w * (float)src[index++];
//         b += w * (float)src[index];

//         sum += w;
//       }
//     }
//     r /= sum;
//     g /= sum;
//     b /= sum;

//     dst[dst_index++] = (uchar)clamp((int)round(r), 0, 255);
//     dst[dst_index++] = (uchar)clamp((int)round(g), 0, 255);
//     dst[dst_index] = (uchar)clamp((int)round(b), 0, 255);
//   }
// }

// #define size 5
// __kernel void pyrdown(__global uchar *src, __global uchar *dst, int width,
//                       int height) {
//   int x = get_global_id(0);
//   int y = get_global_id(1);

//   if (x < width && y < height) {

//     int dst_index = ((int)round(((float)y / 2))) * (width / 2) * 3 +
//                     ((int)round(((float)x / 2))) * 3;

//     // float kenel[size][size] = {
//     //     {1.f, 4.f, 6.f, 4.f, 1.f},
//     //     {4.f, 16.f, 24.f, 16.f, 4.f},
//     //     {6.f, 24.f, 36.f, 24.f, 6.f},
//     //     {4.f, 16.f, 24.f, 16.f, 4.f},
//     //     {1.f, 4.f, 6.f, 4.f, 1.f}
//     // };
// float kenel[size][size] = {
//       {0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625},
//       {0.015625, 0.0625, 0.09375, 0.0625, 0.015625},
//       {0.0234375, 0.09375, 0.140625, 0.09375, 0.0234375},
//       {0.015625, 0.0625, 0.09375, 0.0625, 0.015625},
//       {0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625}};

//     float r = 0.0f, g = 0.0f, b = 0.0f;
//     float sum = 0.0f;
//     int px, py;

// #pragma unroll
//     for (int ky = -2; ky <= 2; ky++) {
//       py = y + ky;
//       for (int kx = -2; kx <= 2; kx++) {
//         px = x + kx;

//         // Apply border reflection for out-of-bound indices
//         if (px < 0)
//           px = -px;
//         else if (px >= width)
//           px = (2 * width) - px - 2;
//         if (py < 0)
//           py = -py;
//         else if (py >= height)
//           py = (2 * height) - py - 2;

//         float w = kenel[ky + 2][kx + 2];
//         // w = w / 256.f;

//         int index = py * width * 3 + px * 3;

//         r += (w * (float)src[index++]);
//         g += (w * (float)src[index++]);
//         b += (w * (float)src[index]);

//         sum += w;
//       }
//     }
//     r /= sum;
//     g /= sum;
//     b /= sum;
//     dst[dst_index++] = (uchar)clamp((int)round(r), 0, 255);
//     dst[dst_index++] = (uchar)clamp((int)round(g), 0, 255);
//     dst[dst_index] = (uchar)clamp((int)round(b), 0, 255);
//   }
// }

// #define size 5
// __kernel void pyrdown(__global uchar *src, __global uchar *dst, int width,
//                       int height) {
//     int x = get_global_id(0);
//     int y = get_global_id(1);

//     if (x < width && y < height) {
//         int src_index = y * width * 3 + x * 3;
//         int dst_index = ((int)round(((float)y / 2))) * (width / 2) * 3 + ((int)round(((float)x / 2))) * 3;

//         float kenel[size][size] = {
//             {1.f, 4.f, 6.f, 4.f, 1.f},
//             {4.f, 16.f, 24.f, 16.f, 4.f},
//             {6.f, 24.f, 36.f, 24.f, 6.f},
//             {4.f, 16.f, 24.f, 16.f, 4.f},
//             {1.f, 4.f, 6.f, 4.f, 1.f}
//         };

//         float r = 0.0f, g = 0.0f, b = 0.0f;
//         float sum = 0.0f;
//         int px, py;

// #pragma unroll
//         for (int ky = -2; ky <= 2; ky++) {
//             py = y + ky;
//             for (int kx = -2; kx <= 2; kx++) {
//                 px = x + kx;

//                 // Apply border reflection for out-of-bound indices
//                 if (px < 0)
//                     continue;
//                 else if (px >= width)
//                     continue;
//                 if (py < 0 || py>=height)
//                 continue;
                    

//                 float w = kenel[ky + 2][kx + 2];
//                 w = w / 256.f;

//                 int index = py * width * 3 + px * 3;

//                 r += (w * (float)src[index++]);
//                 g += (w * (float)src[index++]);
//                 b += (w * (float)src[index]);

//                 sum += w;
//             }
//         }
//         r /= sum;
//         g /= sum;
//         b /= sum;
        
//         dst[dst_index++] = (uchar)round(clamp((r), 0.0f, 255.0f));
// 	dst[dst_index++] = (uchar)round (clamp((g), 0.0f, 255.0f));
// 	dst[dst_index++] = (uchar)round(clamp((b), 0.0f, 255.0f));

//        // dst[dst_index++] = (uchar)round(clamp((g), 0, 255));
//        // dst[dst_index] = (uchar)round(clamp((b), 0, 255));
//     }
// }


/*__kernel void pyrdown1(__global uchar* input, __global uchar* output, const
int width, const int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width && y < height) {
        int in_idx = y * width * 3 + x * 3;
        int out_idx = (y * 0.5) * (width * 1.5) + (x * 1.5);

        // Apply a Gaussian filter to the input image
        float kenel[5] = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};
        float sum = 0.0f;
        float val[3] = {0.0f};
        #pragma unroll
        for (int i = -2; i <= 2; i++) {
            if(y + i >= 0 && y + i < height){
                for (int j = -2; j <= 2; j++) {
                    if ((x + j >= 0 && x + j < width)) {
                        int idx = (y + i) * width * 3 + (x + j) * 3;

                        val[0] += kenel[i + 2] * kenel[j + 2] * (float)
input[idx]; val[1] += kenel[i + 2] * kenel[j + 2] * (float) input[idx + 1];
                        val[2] += kenel[i + 2] * kenel[j + 2] * (float)
input[idx + 2];

                        sum += kenel[i + 2] * kenel[j + 2];
                    }
                }
            }
        }
        val[0] /= sum;
        output[out_idx] = (uchar) (val[0] > 255.0f ? 255.0f : val[0]); // Clamp
the value to 255 if it exceeds the maximum value val[1] /= sum; output[out_idx +
1] = (uchar) (val[1] > 255.0f ? 255.0f : val[1]); val[2] /= sum; output[out_idx
+ 2] = (uchar) (val[2] > 255.0f ? 255.0f : val[2]);
    }
}*/

/*__kernel void pyrdown1(__global uchar* input, __global uchar* output, const
int width, const int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width && y < height) {
        int in_idx = y * width * 3 + x * 3;
        int out_idx = (y / 2) * (width / 2) * 3 + (x / 2) * 3;

        // Copy the pixel value to the output image
        for (int i = 0; i < 3; i++) {
            output[out_idx + i] = input[in_idx + i];
        }

        // Apply a Gaussian filter to the input image
        float kenel[5] = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};
        float sum = 0.0f;
        float val[3] = {0.0f};
        for (int i = -2; i <= 2; i++) {
            for (int j = -2; j <= 2; j++) {
                if ((x + j >= 0 && x + j < width) && (y + i >= 0 && y + i <
height)) { int idx = (y + i) * width * 3 + (x + j) * 3; for (int k = 0; k < 3;
k++) { val[k] += kenel[i + 2] * kenel[j + 2] * (float) input[idx + k];
                    }
                    sum += kenel[i + 2] * kenel[j + 2];
                }
            }
        }
        for (int k = 0; k < 3; k++) {
            val[k] /= sum;
            output[out_idx + k] = (uchar) (val[k] > 255.0f ? 255.0f : val[k]);
// Clamp the value to 255 if it exceeds the maximum value
        }
    }
}*/
