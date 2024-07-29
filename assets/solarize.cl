#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__kernel void solarize(__global const uchar4* inputImage,
                       __global uchar4* resultImage,
                       const float rThresh,
                       const float gThresh,
                       const float bThresh,
                       const int numRows,
                       const int numCols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < numCols && y < numRows) {
        int index = y * numCols + x;

        // Read pixel from input image
        uchar4 pixel = inputImage[index];

        // Extract RGB values and normalize to [0, 1]
        float4 rgb = (float4)(pixel.x, pixel.y, pixel.z, 255) / 255.0f;

        // Apply solarize effect
        if (rgb.x < rThresh)
            rgb.x = 1.0f - rgb.x;
        if (rgb.y < gThresh)
            rgb.y = 1.0f - rgb.y;
        if (rgb.z < bThresh)
            rgb.z = 1.0f - rgb.z;

        // Convert back to uchar range and write to output image
        resultImage[index] = (uchar4)(rgb.x * 255.0f, rgb.y * 255.0f, rgb.z * 255.0f, 255);
    }
}
