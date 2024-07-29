__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void nearestNeighborFilter(__read_only image2d_t inputImage, __write_only image2d_t outputImage, int filterSize, int numRows, int numCols) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= numCols || y >= numRows) {
        return;
    }

    float4 nearest = (float4)(0.0, 0.0, 0.0, 0.0);
    float nearestAvg = 1.0;

    for (int i = 0; i < filterSize; ++i) {
        for (int j = 0; j < filterSize; ++j) {
            int2 neighborPos = (int2)(x + i, y + j);
            float4 rgb = read_imagef(inputImage, sampler, neighborPos);
            float aRgb = (rgb.x + rgb.y + rgb.z) / 3.0f;
            float aNearest = (nearest.x + nearest.y + nearest.z) / 3.0f;
            float difference = aRgb - aNearest;
            if (difference < nearestAvg) {
                nearest = rgb;
                nearestAvg = difference;
            }
        }
    }

    write_imagef(outputImage, (int2)(x, y), nearest);
}
