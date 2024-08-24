// Kernel to perform no operation (pass-through)
__kernel void processImage(__read_only image2d_t inputImage,
                           __write_only image2d_t resultImage)
{
    // Get the coordinates of the current pixel
    int2 pixelCoords = (int2)(get_global_id(0), get_global_id(1));

    // Read the pixel from the input image
    float4 pixel = read_imagef(inputImage, pixelCoords);

    // Write the pixel to the result image
    write_imagef(resultImage, pixelCoords, pixel);
}
