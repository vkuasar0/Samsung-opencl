__kernel void rgb2(__global const uchar *inputImage, __global uchar *outputImage, const int width, const int height) {
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= width || y >= height) {
        return;
    }

    const int idx = (y * width + x) * 3;
    const uchar R = inputImage[idx];
    const uchar G = inputImage[idx + 1];
    const uchar B = inputImage[idx + 2];

    // Convert RGB to YCbCr
    const float Y = 0.299f * R + 0.587f * G + 0.114f * B;
    const float Cr = (R - Y) * 0.713f + 128.0f;
    const float Cb = (B - Y) * 0.564f + 128.0f;

    // Clamp and store the results
    outputImage[idx] = (uchar)clamp((int)round(Y), 0, 255);
    outputImage[idx + 1] = (uchar)clamp((int)round(Cr), 0, 255);
    outputImage[idx + 2] = (uchar)clamp((int)round(Cb), 0, 255);
}
