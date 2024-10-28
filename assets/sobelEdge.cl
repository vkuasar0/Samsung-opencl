__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void sobelEdge(__read_only image2d_t srcImage, __write_only image2d_t dstImage) {
    const int2 pos = {get_global_id(0), get_global_id(1)};
    
    float Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    
    float Gy[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}
    };
    
    float4 sumX = (float4)(0.0f);
    float4 sumY = (float4)(0.0f);
    
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            int2 coord = pos + (int2)(i, j);
            coord.x = clamp(coord.x, 0, get_image_width(srcImage) - 1);
            coord.y = clamp(coord.y, 0, get_image_height(srcImage) - 1);

            // Read color as uint4 and convert to float4 for calculations
            uint4 color_uint = read_imageui(srcImage, sampler, coord);
            float4 color = (float4)(color_uint.x, color_uint.y, color_uint.z, color_uint.w);

            // Apply Sobel filters
            sumX += Gx[i + 1][j + 1] * color;
            sumY += Gy[i + 1][j + 1] * color;
        }
    }

    float4 magnitude = sqrt(sumX * sumX + sumY * sumY);
    magnitude.w = 255.0f; // Set alpha to fully opaque

    uint4 output_pixel = (uint4)(
        clamp((int)(magnitude.x + 0.5f), 0, 255),
        clamp((int)(magnitude.y + 0.5f), 0, 255),
        clamp((int)(magnitude.z + 0.5f), 0, 255),
        255 // Ensure alpha is set to maximum
    );

    write_imageui(dstImage, pos, output_pixel);
}