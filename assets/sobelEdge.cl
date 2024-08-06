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
            
            float4 color = read_imagef(srcImage, sampler, coord);
            sumX += Gx[i + 1][j + 1] * color;
            sumY += Gy[i + 1][j + 1] * color;
        }
    }
    
    float4 magnitude = sqrt(sumX * sumX + sumY * sumY);
    magnitude.w = 1.0f; // Ensure the alpha channel is set to 1.0
    write_imagef(dstImage, pos, magnitude);
}
