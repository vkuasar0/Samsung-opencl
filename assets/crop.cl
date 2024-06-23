/*__kernel void crop(__global const uchar *input, const int width,
                   const int height, __global uchar *output,
                   const int roi_width, const int roi_height, const int x,
                   const int y) {
  int row = get_global_id(1);
  int col = get_global_id(0);
  int channels = 3;
  if (row < roi_height && col < roi_width) {
    int input_index = (row + y) * width * channels + (col + x) * channels;
    int output_index = row * roi_width * channels + col * channels;

    uchar4 pixel;
    __global const uchar *input_ptr = input + input_index;
    __global uchar *output_ptr = output + output_index;

    for (int c = 0; c < channels; c += 4) {
      pixel = vload4(c, input_ptr);
      vstore4(pixel, c, output_ptr);
    }
  }
}
*/
__kernel void crop(__global uchar* input_img,  int input_width, int input_height, __global uchar* output_img,int output_width, int output_height, int x_offset, int y_offset)
{
    int global_x = get_global_id(0);
    int global_y = get_global_id(1);
    
    if(global_x >= output_width || global_y >= output_height)
        return;
    
    int input_x = global_x + x_offset;
    int input_y = global_y + y_offset;
    
    if(input_x >= input_width || input_y >= input_height)
        return;
    
    int input_index = (input_y * input_width + input_x) * 3;
    int output_index = (global_y * output_width + global_x) * 3;
    
    output_img[output_index + 0] = input_img[input_index + 0];
    output_img[output_index + 1] = input_img[input_index + 1];
    output_img[output_index + 2] = input_img[input_index + 2];
}


