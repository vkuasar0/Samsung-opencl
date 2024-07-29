#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

__kernel void pixellate(__global const uchar4* inputImage,
                        __global uchar4* resultImage,
                        const int filterSize,
                        const int numRows,
                        const int numCols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    // Calculate the top-left corner of the current block
    int startX = x - (x % filterSize);
    int startY = y - (y % filterSize);

    if (startX < numCols && startY < numRows) {
        // Accumulate the sum of colors within the block
        float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

        // Count the number of pixels in the block
        int count = 0;

        for (int i = 0; i < filterSize && startX + i < numCols; i++) {
            for (int j = 0; j < filterSize && startY + j < numRows; j++) {
                int index = (startY + j) * numCols + (startX + i);
                uchar4 pixel = inputImage[index];

                // Accumulate colors
                sum += (float4)(pixel.x, pixel.y, pixel.z, 0.0f);
                count++;
            }
        }

        // Calculate the average color
        float4 avgColor = sum / count;

        // Assign the average color to all pixels in the block
        for (int i = 0; i < filterSize && startX + i < numCols; i++) {
            for (int j = 0; j < filterSize && startY + j < numRows; j++) {
                int index = (startY + j) * numCols + (startX + i);
                resultImage[index] = (uchar4)(avgColor.x, avgColor.y, avgColor.z, 255);
            }
        }
    }
}
