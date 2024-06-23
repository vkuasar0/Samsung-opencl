/*__kernel void downsize(__global uchar* input_image,
__global uchar* output_image,
const int input_width,
const int input_height,    const int output_width,
                           const int output_height) {
const int output_size = output_width * output_height;
const int input_size = input_width * input_height;
// Compute the scaling factors for width and height
const float scale_x = (float) output_width / (float) input_width;
const float scale_y = (float) output_height / (float) input_height;
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
const int in_x = (int) round(out_x / scale_x);
const int in_y = (int) round(out_y / scale_y);
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

/*
__kernel void downsize(__global uchar* input, __global uchar* output, const int input_width, const int input_height, const int output_width, const int output_height)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= output_width || y >= output_height) {
        return;
    }
    int input_x;
    int input_y;
    const float input_x_float = x * (float)input_width / output_width;
    const float input_y_float = y * (float)input_height / output_height;
    if (input_x_float>=0 && input_y_float>=0)
   {input_x = (int)(input_x_float);
    input_y = (int)(input_y_float);
   }
    else 
    {input_x = (int)(input_x_float+1) ;
    input_y = (int)(input_y_float+1);
    }
        const int input_idx = (input_y * input_width + input_x) * 3;
    const int output_idx = (y * output_width + x) * 3;

    output[output_idx] = input[input_idx];
    output[output_idx + 1] = input[input_idx + 1];
    output[output_idx + 2] = input[input_idx + 2 ];
}*/


__kernel void downsize(__global uchar* inImg, __global uchar* outImg, int
inWidth, int inHeight, int outWidth, int outHeight)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if(x >= outWidth || y >= outHeight) return;


    int inX = round(((float)(x * inWidth + (outWidth/2))) / outWidth);
    int inY = round(((float)(y * inHeight + (outHeight/2))) / outHeight);

    int inIdx = (inY * inWidth + inX) * 3;
    int outIdx = (y * outWidth + x) * 3;

    outImg[outIdx + 0] = inImg[inIdx + 0]; // red channel
    outImg[outIdx + 1] = inImg[inIdx + 1]; // green channel
    outImg[outIdx + 2] = inImg[inIdx + 2]; // blue channel
}


