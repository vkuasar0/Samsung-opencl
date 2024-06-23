#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

__kernel void absdiff_opt(__global uchar3 *image1, __global uchar3 *image2,
                          __global uchar3 *output, int width, int height) {
  int gid_x = get_global_id(0);
  int gid_y = get_global_id(1);
  int idx = gid_y * width + gid_x;

  if (gid_x < width && gid_y < height) {
    uchar3 val1 = image1[idx];
    uchar3 val2 = image2[idx];
    float3 sum = (float3)(abs(val1.s0 - val2.s0), abs(val1.s1 - val2.s1),
                          abs(val1.s2 - val2.s2));
    uchar3 out = convert_uchar3(clamp(sum, 0.0f, 255.0f));
    output[idx] = out;
  }
}

__kernel void add_opt(__global uchar3 *image1, __global uchar3 *image2,
                      __global uchar3 *output, int width, int height) {
  // local memory
  __local uchar4 local_image1[BLOCK_SIZE_X + 2][BLOCK_SIZE_Y + 2];
  __local uchar4 local_image2[BLOCK_SIZE_X + 2][BLOCK_SIZE_Y + 2];

  int2 global_coord = (int2)(get_global_id(1), get_global_id(0));
  int2 local_coord = (int2)(get_local_id(1), get_local_id(0));
  int2 block_coord = (int2)(get_group_id(1), get_group_id(0));

  int local_width = BLOCK_SIZE_X + 2;
  int local_height = BLOCK_SIZE_Y + 2;

  int2 local_image_coord = local_coord + 1;
  int2 global_image_coord = block_coord * BLOCK_SIZE_X + local_image_coord;

  // Load input images into local memory
  uchar3 image1_pixel =
      image1[global_image_coord.x * width + global_image_coord.y];
  uchar3 image2_pixel =
      image2[global_image_coord.x * width + global_image_coord.y];

  local_image1[local_image_coord.x][local_image_coord.y].x = image1_pixel.x;
  local_image1[local_image_coord.x][local_image_coord.y].y = image1_pixel.y;
  local_image1[local_image_coord.x][local_image_coord.y].z = image1_pixel.z;
  local_image1[local_image_coord.x][local_image_coord.y].w = 0;

  local_image2[local_image_coord.x][local_image_coord.y].x = image2_pixel.x;
  local_image2[local_image_coord.x][local_image_coord.y].y = image2_pixel.y;
  local_image2[local_image_coord.x][local_image_coord.y].z = image2_pixel.z;
  local_image2[local_image_coord.x][local_image_coord.y].w = 0;

  // Synchronize to ensure input images are fully loaded
  barrier(CLK_LOCAL_MEM_FENCE);

  uchar3 pixel1 = image1[global_coord.x * width + global_coord.y];
  uchar3 pixel2 = image2[global_coord.x * width + global_coord.y];

  // clamp(x, minval, maxval)
  // clamp returns min (min( x, minval), maxval)
  uchar4 result;
  // result.x = (uchar)((int)pixel1.x + (int)pixel2.x > 255 ? 255 :
  // (int)pixel1.x + (int)pixel2.x); result.y = (uchar)((int)pixel1.y +
  // (int)pixel2.y > 255 ? 255 : (int)pixel1.y + (int)pixel2.y); result.z =
  // (uchar)((int)pixel1.z + (int)pixel2.z > 255 ? 255 : (int)pixel1.z +
  // (int)pixel2.z);
  result.x = clamp((int)pixel1.x + (int)pixel2.x, 0, 255);
  result.y = clamp((int)pixel1.y + (int)pixel2.y, 0, 255);
  result.z = clamp((int)pixel1.z + (int)pixel2.z, 0, 255);
  result.w = 0;

  output[global_coord.x * width + global_coord.y] =
      (uchar3)(result.x, result.y, result.z);

  // Synchronize to ensure output image is fully stored
  barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel void add_images(__global uchar *image1, __global uchar *image2,
                         __global uchar *output, int width, int height) {
  int row = get_global_id(0);
  int col = get_global_id(1);
  int idx = (row * width + col) * 3;

  // Check if indices are within bounds
  if (row >= height || col >= width)
    return;

  // Compute pixel values for output image
  for (int i = 0; i < 3; i++) {

    int val1 = image1[idx + i];
    int val2 = image2[idx + i];
    int sum = val1 + val2;
    output[idx + i] = (uchar)(sum > 255 ? 255 : sum);
  }
}

__kernel void sub_images(__global uchar *image1, __global uchar *image2,
                         __global uchar *output, int width, int height) {
  int row = get_global_id(0);
  int col = get_global_id(1);
  int idx = (row * width + col) * 3;

  // Check if indices are within bounds
  if (row >= height || col >= width)
    return;

  // Compute pixel values for output image
  for (int i = 0; i < 3; i++) {
    int val1 = image1[idx + i];
    int val2 = image2[idx + i];
    int diff = val1 - val2;
    output[idx + i] = (uchar)(diff < 0 ? 0 : diff);
  }
}

__kernel void absdiff_images(__global uchar *image1, __global uchar *image2,
                             __global uchar *output, int width, int height) {
  int row = get_global_id(0);
  int col = get_global_id(1);
  int idx = (row * width + col) * 3;

  // Check if indices are within bounds
  if (row >= height || col >= width)
    return;

  // Compute pixel values for output image
  for (int i = 0; i < 3; i++) {
    int val1 = image1[idx + i];
    int val2 = image2[idx + i];
    int diff = abs(val1 - val2);
    output[idx + i] = (uchar)diff;
  }
}

__kernel void mul_images(__global uchar *image1, int multiplier,
                         __global uchar *output, int width, int height) {
  int row = get_global_id(0);
  int col = get_global_id(1);
  int idx = (row * width + col) * 3;

  // Check if indices are within bounds
  if (row >= height || col >= width)
    return;

  // Compute pixel values for output image
  for (int i = 0; i < 3; i++) {
    int val1 = image1[idx + i];

    int prod = (val1 * multiplier);
    output[idx + i] = (uchar)(prod > 255 ? 255 : prod);
  }
}

__kernel void div_image(__global uchar *image1, int divisor,
                        __global uchar *output, int width, int height) {
  int row = get_global_id(0);
  int col = get_global_id(1);
  int idx = (row * width + col) * 3;

  // Check if indices are within bounds
  if (row >= height || col >= width)
    return;

  // Compute pixel values for output image
  for (int i = 0; i < 3; i++) {
    int val1 = image1[idx + i];

    int prod = (val1 / divisor);
    output[idx + i] = (uchar)(prod < 0 ? 0 : prod);
  }
}

__kernel void equal_images(__global uchar *image1, __global uchar *image2,
                           __global uchar *output, int width, int height) {
  int row = get_global_id(0);
  int col = get_global_id(1);
  int idx = (row * width + col) * 3;

  // Check if indices are within bounds
  if (row >= height || col >= width)
    return;

  // Compute pixel values for output image
  for (int i = 0; i < 3; i++) {
    if (image1[idx + i] == image2[idx + i])
      output[idx + i] = image1[idx + i];
    else
      output[idx + i] = 0;
  }
}

__kernel void count_nonzero_pixels(__global uchar *image, __global int *count_r,
                                   __global int *count_g, __global int *count_b,
                                   const int width, const int height) {

  int row = get_global_id(0);
  int col = get_global_id(1);

  if (row < height && col < width) {
    int index = (row * width + col) * 3;

    if (image[index] > 0) {
      atomic_inc(count_r);
    }
    if (image[index + 1] > 0) {
      atomic_inc(count_g);
    }
    if (image[index + 2] > 0) {
      atomic_inc(count_b);
    }
  }
}

__kernel void split_image(__global uchar *input, __global uchar *output1,
                          __global uchar *output2, __global uchar *output3,
                          int width, int height) {
  int row = get_global_id(0);
  int col = get_global_id(1);

  // Check if indices are within bounds
  if (row >= height || col >= width)
    return;

  int idx = (row * width + col) * 3;

  // Copy pixel values from input to output channels
  output1[idx / 3] = input[idx];
  output2[idx / 3] = input[idx + 1];
  output3[idx / 3] = input[idx + 2];
}

__kernel void merge_kernel(__global uchar *input1, __global uchar *input2,
                           __global uchar *input3, __global uchar *output,
                           int width, int height) {
  int row = get_global_id(0);
  int col = get_global_id(1);

  // Check if indices are within bounds
  if (row >= height || col >= width)
    return;

  int idx = (row * width + col) * 3;

  // Copy pixel values from input1, input2 and input3 to output
  output[idx] = input1[row * width + col];
  output[idx + 1] = input2[row * width + col];
  output[idx + 2] = input3[row * width + col];
}

__kernel void mean_images(__global uchar *image1, __global uchar *image2,
                          __global uchar *output, int width, int height) {
  int row = get_global_id(0);
  int col = get_global_id(1);
  int idx = (row * width + col) * 3;

  // Check if indices are within bounds
  if (row >= height || col >= width)
    return;

  // Compute pixel values for output image
  for (int i = 0; i < 3; i++) {
    output[idx + i] = (uchar)((image1[idx + i] + image2[idx + i]) / 2);
  }
}

__kernel void crop(__global const uchar *input, const int width,
                   const int height, __global uchar *output,
                   const int roi_width, const int roi_height, const int x,
                   const int y) {
  int row = get_global_id(1);
  int col = get_global_id(0);
  int channels = 3;
  if (row < roi_height && col < roi_width) {
    int input_index = (row + y) * width * channels + (col + x) * channels;
    int output_index = row * roi_width * channels + col * channels;
    for (int c = 0; c < channels; c++) {
      output[output_index + c] = input[input_index + c];
    }
  }
}

__kernel void gray(__global const uchar *input, __global uchar *output,
                   int width, int height) {
  int x = get_global_id(0);
  int y = get_global_id(1);

  int index = y * width + x;

  uchar b = input[index * 3];
  uchar g = input[index * 3 + 1];
  uchar r = input[index * 3 + 2];

  const int COEFF_R = 19595; // 0.299 * 2^16 + 0.5
  const int COEFF_G = 38470; // 0.587 * 2^16 + 0.5
  const int COEFF_B = 7471;  // 0.114 * 2^16 + 0.5

  int gray = (COEFF_R * r + COEFF_G * g + COEFF_B * b + 32768) >> 16;

  output[index] = (uchar)gray;
}

__kernel void hsv(__global const uchar *input, __global uchar *output,
                  int width, int height) {

  int x = get_global_id(0);
  int y = get_global_id(1);
  int index = y * width + x;

  uchar b = input[index * 3];
  uchar g = input[index * 3 + 1];
  uchar r = input[index * 3 + 2];

  float min_val = min(b, min(g, r));
  float max_val = max(b, max(g, r));
  float delta = max_val - min_val;

  float h = 0.0f;
  float s = 0.0f;
  float v = max_val / 255.0f;

  if (max_val > 0.0f) {
    s = delta / max_val;
    if (max_val == r) {
      h = 60.0f * (g - b) / delta;
    } else if (max_val == g) {
      h = 60.0f * (2.0f + (b - r) / delta);
    } else {
      h = 60.0f * (4.0f + (r - g) / delta);
    }
    if (h < 0.0f)
      h += 360.0f;
  }

  float hue_norm = h / 2.0f;   // normalize hue to range [0, 180]
  float sat_norm = s * 255.0f; // normalize saturation to range [0, 255]
  float val_norm = v * 255.0f; // normalize value to range [0, 255]

  uchar hue = (uchar)round(hue_norm);
  uchar sat = (uchar)round(sat_norm);
  uchar val = (uchar)round(val_norm);

  output[index * 3] = hue;
  output[index * 3 + 1] = sat;
  output[index * 3 + 2] = val;
}

__kernel void reshape_image(__global const uchar *input_image,
                            __global uchar *output_image, const int input_width,
                            const int input_height, const int output_width,
                            const int output_height) {
  const int input_channels = 3;  // RGB channels
  const int output_channels = 3; // RGB channels
  const int input_size = input_width * input_height * input_channels;
  const int output_size = output_width * output_height * output_channels;

  int x = get_global_id(0);
  int y = get_global_id(1);

  if (x < output_width && y < output_height) {
    int input_x = x * input_width / output_width;
    int input_y = y * input_height / output_height;

    for (int c = 0; c < output_channels; c++) {
      int input_idx =
          input_y * input_width * input_channels + input_x * input_channels + c;
      int output_idx =
          y * output_width * output_channels + x * output_channels + c;
      output_image[output_idx] = input_image[input_idx];
    }
  }
}