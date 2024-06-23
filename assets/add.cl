__kernel void add_images(__read_only image2d_t im1, __read_only image2d_t im2, __write_only image2d_t im3) {
    int2 gid = (int2)(get_global_id(0), get_global_id(1));
    int width = get_image_width(im1);
    int height = get_image_height(im1);

    // Bounds checking
    if (gid.x >= width || gid.y >= height) {
        return;
    }

    const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    const uint4 pin1 = read_imageui(im1, smp, gid);
    const uint4 pin2 = read_imageui(im2, smp, gid);
    const uint4 sum = clamp(pin1 + pin2, (uint4)0, (uint4)255);
    write_imageui(im3, gid, sum);
}
