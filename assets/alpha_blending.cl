__kernel void alpha_blending(__read_only image2d_t img1, __read_only image2d_t img2, __write_only image2d_t output, float alpha) {
    int2 pos = (int2)(get_global_id(0), get_global_id(1));
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    uint4 pixel1 = read_imageui(img1, sampler, pos);
    uint4 pixel2 = read_imageui(img2, sampler, pos);

    float4 float_pixel1 = (float4)(pixel1.x, pixel1.y, pixel1.z, pixel1.w);
    float4 float_pixel2 = (float4)(pixel2.x, pixel2.y, pixel2.z, pixel2.w);

    float4 blended_pixel = alpha * float_pixel1 + (1.0f - alpha) * float_pixel2;

    uint4 output_pixel = (uint4)(
        clamp((int)(blended_pixel.x + 0.5f), 0, 255),
        clamp((int)(blended_pixel.y + 0.5f), 0, 255),
        clamp((int)(blended_pixel.z + 0.5f), 0, 255),
        255 // Assuming fully opaque output
    );

    write_imageui(output, pos, output_pixel);
}
