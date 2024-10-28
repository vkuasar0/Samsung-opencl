__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void median_filter(__read_only image2d_t input_image, 
                            __write_only image2d_t output_image) 
{
    // Get image dimensions directly
    int width = get_image_width(input_image);
    int height = get_image_height(input_image);

    // Get the row and column of the pixel
    int2 pos = {get_global_id(0), get_global_id(1)};

    // Size of the filter window (3x3)
    const int filter_size = 3;
    const int filter_half = filter_size / 2;

    // Temporary array to hold the values in the filter window
    uchar window[filter_size * filter_size];
    int count = 0;

    // Iterate over the filter window
    for (int i = -filter_half; i <= filter_half; i++) {
        for (int j = -filter_half; j <= filter_half; j++) {
            // Compute the coordinates of the neighboring pixel
            int2 neighbor_pos = {pos.x + j, pos.y + i};

            // Check for out-of-bound coordinates
            if (neighbor_pos.x >= 0 && neighbor_pos.x < width && 
                neighbor_pos.y >= 0 && neighbor_pos.y < height) {
                // Read the pixel value and store in window array
                uint4 pixel = read_imageui(input_image, sampler, neighbor_pos);
                window[count++] = pixel.x; // Assuming single channel for simplicity
            } else {
                // For out-of-bound pixels, add a default value (e.g., 0)
                window[count++] = 0;
            }
        }
    }

    // Sort the values in the window array to find the median
    for (int i = 0; i < count - 1; i++) {
        for (int j = i + 1; j < count; j++) {
            if (window[i] > window[j]) {
                uchar temp = window[i];
                window[i] = window[j];
                window[j] = temp;
            }
        }
    }

    // Write the median value to the output image
    write_imageui(output_image, pos, (uint4)(window[count / 2], 0, 0, 255));
}