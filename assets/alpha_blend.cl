__kernel void alpha_blending(__global const uchar4* img1, __global const uchar4* img2, __global uchar4* output, float alpha, unsigned int width, unsigned int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width && y < height) {
        int index = y * width + x;

        uchar4 pixel1 = img1[index];
        uchar4 pixel2 = img2[index];

        // Perform alpha blending
        output[index].x = (uchar)(alpha * pixel1.x + (1.0f - alpha) * pixel2.x);
        output[index].y = (uchar)(alpha * pixel1.y + (1.0f - alpha) * pixel2.y);
        output[index].z = (uchar)(alpha * pixel1.z + (1.0f - alpha) * pixel2.z);
        output[index].w = 255; // Assuming fully opaque output image
    }
}
