__kernel void mul(__global const uchar4* image1, __global const uchar4* image2, __global uchar4* output, int width, int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width && y < height) {
        int index = y * width + x;
        uchar4 pixel1 = image1[index];
        uchar4 pixel2 = image2[index];

        uchar4 result;
        result.x = clamp(pixel1.x * pixel2.x / 255, 0, 255);
        result.y = clamp(pixel1.y * pixel2.y / 255, 0, 255);
        result.z = clamp(pixel1.z * pixel2.z / 255, 0, 255);
        result.w = clamp(pixel1.w * pixel2.w / 255, 0, 255);

        output[index] = result;
    }
}
