__kernel void gray_bgr(__global const uchar* inputImage, __global uchar* outputImage,
                       int width, int height, int channels) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < width && y < height) {
        int index = (y * width + x) * channels;
        uchar b = inputImage[index];
        uchar g = inputImage[index + 1];
        uchar r = inputImage[index + 2];

        // Calculate grayscale value using the luminosity method
        float grayFloat = 0.299f * r + 0.587f * g + 0.114f * b;
        uchar gray = convert_uchar_sat(grayFloat); // Convert to uchar with saturation

        // Write grayscale value to output image
        outputImage[y * width + x] = gray;
    }
}