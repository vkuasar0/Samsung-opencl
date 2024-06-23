/*sssssssssssssssssssssssssssssssssssssssssssssssssssssssss
__kernel void count_nonzero_pixels(__global uchar* image, 
                                    __global int* count_r, 
                                    __global int* count_g, 
                                    __global int* count_b, 
                                    const int width, 
                                    const int height) {

   const int gid_x = get_global_id(0);
  const int gid_y = get_global_id(1);
   
    
    if (gid_x >= width || gid_y >= height)
    return;
 
  const int input_idx = (gid_y * width + gid_x) * 3;
  

  if (image[input_idx]>0)
  	atomic_add(count_r,1);

   if (image[input_idx+1]>0)
  	atomic_add(count_g,1);

   if (image[input_idx+2]>0)
  	atomic_add(count_b,1);

  
}
*/

__kernel void count_nonzero_pixels_r(__global uchar* image, __global int* count_r, const int width, const int height) {
   const int gid_x = get_global_id(0);
  const int gid_y = get_global_id(1);
    int idx = (gid_y * width + gid_x)*3;
    if (gid_x < width && gid_y < height) {
        if (image[idx] > 0) {
            atomic_add(count_r,1);
        }
    }
}

__kernel void count_nonzero_pixels_g(__global uchar* image, __global int* count_g, const int width, const int height) {
    const int gid_x = get_global_id(0);
  const int gid_y = get_global_id(1);
    int idx = (gid_y * width + gid_x)*3;
    if (gid_x < width && gid_y < height) {
        if (image[idx+1] > 0) {
            atomic_add(count_g,1);
        }
    }
}
__kernel void count_nonzero_pixels_b(__global uchar* image, __global int* count_b, const int width, const int height) {
    const int gid_x = get_global_id(0);
  const int gid_y = get_global_id(1);
    int idx = (gid_y * width + gid_x)*3;
    if (gid_x < width && gid_y < height) {
        if (image[idx+2] > 0) {
            atomic_add(count_b,1);
        }
    }
}
