__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void alpha_blending(__read_only image2d_t img1, __read_only image2d_t img2, __write_only image2d_t output, float alpha) {
    int2 pos = {get_global_id(0), get_global_id(1)};

    // Read pixel values from img1 and img2 at the current position
    float4 pixel1 = read_imagef(img1, sampler, pos);
    float4 pixel2 = read_imagef(img2, sampler, pos);

    // Ensure alpha is within [0.0, 1.0] to prevent a blank result
    alpha = clamp(alpha, 0.0f, 1.0f);

    // Perform alpha blending
    float4 blended_pixel = alpha * pixel1 + (1.0f - alpha) * pixel2;
    blended_pixel.w = 1.0f; // Set the output to fully opaque

    // Write the blended pixel to the output image
    write_imagef(output, pos, blended_pixel);
}
