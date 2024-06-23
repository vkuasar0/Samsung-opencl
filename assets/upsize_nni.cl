/*__kernel void upsize(__global uchar *input_image, __global uchar *output_image,
                     const int input_width, const int input_height,
                     const int output_width, const int output_height) {

  const int output_size = output_width * output_height;
  const int input_size = input_width * input_height;
  // Compute the scaling factors for width and height
  const float scale_x = (float)input_width / (float)output_width;
  const float scale_y = (float)input_height / (float)output_height;
  // Compute the thread ID
  const int tid_x = get_global_id(0);
  const int tid_y = get_global_id(1);
  // Compute the output pixel coordinates
  const int out_x = tid_x;
  const int out_y = tid_y;
  if (out_x >= output_width || out_y >= output_height) {
    return;
  }
  // Compute the corresponding input pixel coordinates
  const int in_x = (int)(out_x * scale_x + 0.5f);
  const int in_y = (int)(out_y * scale_y + 0.5f);

  // Compute the input pixel index
  const int in_idx = in_y * input_width + in_x;
  // Compute the output pixel index
  const int out_idx = out_y * output_width + out_x;
  // Copy the pixel values from the input to the output
  output_image[3 * out_idx + 0] = input_image[3 * in_idx + 0];
  output_image[3 * out_idx + 1] = input_image[3 * in_idx + 1];
  output_image[3 * out_idx + 2] = input_image[3 * in_idx + 2];
}
*/
/*__kernel void upsize(__global uchar *input_image, __global uchar *output_image,
                     const int input_width, const int input_height,
                     const int output_width, const int output_height) {

  const int output_size = output_width * output_height;
  const int input_size = input_width * input_height;
  // Compute the scaling factors for width and height
  const float scale_x = (float)input_width / (float)output_width;
  const float scale_y = (float)input_height / (float)output_height;
  // Compute the thread ID
  const int tid_x = get_global_id(0);
  const int tid_y = get_global_id(1);
  // Compute the output pixel coordinates
  const int out_x = tid_x;
  const int out_y = tid_y;
  if (out_x >= output_width || out_y >= output_height) {
    return;
  }
  // Compute the corresponding input pixel coordinates using nearest neighbor interpolation
  const int in_x = (int)(out_x * scale_x + 0.5f);
  const int in_y = (int)(out_y * scale_y + 0.5f);
  // Clamp input pixel coordinates to the image boundaries
  const int in_x_clamped = max(0, min(input_width - 1, in_x));
  const int in_y_clamped = max(0, min(input_height - 1, in_y));

  // Compute the input pixel index
  const int in_idx = in_y_clamped * input_width + in_x_clamped;
  // Compute the output pixel index
  const int out_idx = out_y * output_width + out_x;
  // Copy the pixel values from the input to the output
  output_image[3 * out_idx + 0] = input_image[3 * in_idx + 0];
  output_image[3 * out_idx + 1] = input_image[3 * in_idx + 1];
  output_image[3 * out_idx + 2] = input_image[3 * in_idx + 2];
}*/
/*__kernel void upsize(__global uchar* inImg, __global uchar* outImg, int inWidth, int inHeight, int outWidth, int outHeight)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x >= outWidth || y >= outHeight) return;

    int inX = (x * inWidth) / outWidth;
    int inY = (y * inHeight) / outHeight;

    int inIdx = (inY * inWidth + inX) * 3;
    int outIdx = (y * outWidth + x) * 3;

    outImg[outIdx + 0] = inImg[inIdx + 0]; // red channel
    outImg[outIdx + 1] = inImg[inIdx + 1]; // green channel
    outImg[outIdx + 2] = inImg[inIdx + 2]; // blue channel
}
*/
#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

typedef unsigned char T;  // define T as an unsigned char


#define INC(x,l) min(x+1,l-1)

#define noconvert

#if cn != 3
#define loadpix(addr)  *(__global const T *)(addr)
#define storepix(val, addr)  *(__global T *)(addr) = val
#define TSIZE (int)sizeof(T)
#else
#define loadpix(addr)  vload3(0, (__global const T1 *)(addr))
#define storepix(val, addr) vstore3(val, 0, (__global T1 *)(addr))
#define TSIZE (int)sizeof(T1)*cn
#endif



#if cn == 1
#define READ_IMAGE(X,Y,Z)  read_imagef(X,Y,Z).x
#define INTERMEDIATE_TYPE  float
#elif cn == 2
#define READ_IMAGE(X,Y,Z)  read_imagef(X,Y,Z).xy
#define INTERMEDIATE_TYPE  float2
#elif cn == 3
#define READ_IMAGE(X,Y,Z)  read_imagef(X,Y,Z).xyz
#define INTERMEDIATE_TYPE  float3
#elif cn == 4
#define READ_IMAGE(X,Y,Z)  read_imagef(X,Y,Z)
#define INTERMEDIATE_TYPE  float4
#endif

#define __CAT(x, y) x##y
#define CAT(x, y) __CAT(x, y)
//#define INTERMEDIATE_TYPE CAT(float, cn)
#define float1 float

#if depth == 0
#define RESULT_SCALE    255.0f
#elif depth == 1
#define RESULT_SCALE    127.0f
#elif depth == 2
#define RESULT_SCALE    65535.0f
#elif depth == 3
#define RESULT_SCALE    32767.0f
#else
#define RESULT_SCALE    1.0f
#endif

// __kernel void upsize(__global const uchar *input, __global uchar *output,
//                       const int input_width, const int input_height,
//                       const int output_width, const int output_height)
// {
//     const int x = get_global_id(0);
//     const int y = get_global_id(1);

//     if (x >= output_width || y >= output_height) {
//         return;
//     }
//     int input_x;
//     int input_y;
//     const float input_x_float = x * (float)input_width / output_width;
//     const float input_y_float = y * (float)input_height / output_height;
//     if (input_x_float>=0 && input_y_float>=0)
//    {input_x = (int)(input_x_float);
//     input_y = (int)(input_y_float);
//    }
//     else 
//     {input_x = (int)(input_x_float+1) ;
//     input_y = (int)(input_y_float+1);
//     }


//     const int input_idx = (input_y * input_width + input_x) * 3;
//     const int output_idx = (y * output_width + x) * 3;

//     output[output_idx] = input[input_idx];
//     output[output_idx + 1] = input[input_idx + 1];
//     output[output_idx + 2] = input[input_idx + 2 ];
// }

int interpolateChannel(__global uchar* input, int idx00, int idx10, int idx01, int idx11, float dx, float dy)
{
    const float w00 = (1.0 - dx) * (1.0 - dy);
    const float w10 = dx * (1.0 - dy);
    const float w01 = (1.0 - dx) * dy;
    const float w11 = dx * dy;

    const float channelVal = w00 * input[idx00] + w10 * input[idx10] + w01 * input[idx01] + w11 * input[idx11];

    return round(channelVal);
}


__kernel void upsize(__global const uchar *input, __global uchar *output,
                     const int input_width, const int input_height,
                     const int output_width, const int output_height) {
  // const int x = get_global_id(0);
  // const int y = get_global_id(1);

  // if (x >= output_width || y >= output_height)
  //   return;

  // const float input_x_float = x * (float)input_width / output_width;
  // const float input_y_float = y * (float)input_height / output_height;

  // const int input_x = (int)(input_x_float);
  // const int input_y = (int)(input_y_float);

  // const int input_idx = (input_y * input_width + input_x) * 3;

  // output[output_idx] = input[input_idx];
  // output[output_idx + 1] = input[input_idx + 1];
  // output[output_idx + 2] = input[input_idx + 2];


   const int i = get_global_id(0);
    const int j = get_global_id(1);

    if (i >= output_width || j >= output_height)
        return;

    const float scaleX = (input_width -1  ) / (float)(output_width-1 );
    const float scaleY = (input_height-1 ) / (float)(output_height-1 );

    const float ii = i * scaleX;
    const float jj = j * scaleY;

    const int x0 = floor(ii);
    const int y0 = floor(jj);

    const int x1 = min(x0 + 1, input_width - 1);
    const int y1 = min(y0 + 1, input_height - 1);

    const float dx = ii - x0;
    const float dy = jj - y0;

    const int inputIdx00 = (y0 * input_width + x0) * 3;
    const int inputIdx10 = (y0 * input_width + x1) * 3;
    const int inputIdx01 = (y1 * input_width + x0) * 3;
    const int inputIdx11 = (y1 * input_width + x1) * 3;

    const int outputIdx = (j * output_width + i) * 3;

    // Interpolate RGB channels separately
    output[outputIdx + 0] = (interpolateChannel(input, inputIdx00 + 0, inputIdx10 + 0, inputIdx01 + 0, inputIdx11 + 0, dx, dy)); // Red channel
    output[outputIdx + 1] = (interpolateChannel(input, inputIdx00 + 1, inputIdx10 + 1, inputIdx01 + 1, inputIdx11 + 1, dx, dy)); // Green channel
    output[outputIdx + 2] = (interpolateChannel(input, inputIdx00 + 2, inputIdx10 + 2, inputIdx01 + 2, inputIdx11 + 2, dx, dy)); // Blue channel

}


// __kernel void upsize(__global const uchar *input, __global uchar *output,
//                       const int input_width, const int input_height,
//                       const int output_width, const int output_height)
// {
//     const int x = get_global_id(0);
//     const int y = get_global_id(1);

//     if (x >= output_width || y >= output_height)
//         return;


//     const float input_x_float = x * (float)input_width / output_width;
//     const float input_y_float = y * (float)input_height / output_height;

//     const int input_x = (int)(input_x_float);
//     const int input_y = (int)(input_y_float);

//     const int input_idx = (input_y * input_width + input_x) * 3;
//     const int output_idx = (y * output_width + x) * 3;

//     output[output_idx] = input[input_idx];
//     output[output_idx + 1] = input[input_idx + 1];
//     output[output_idx + 2] = input[input_idx + 2];


// }
