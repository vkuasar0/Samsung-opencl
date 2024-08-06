__kernel void median_filter(__global const uchar *input_image, 
                            __global uchar *output_image, 
                            const int width, 
                            const int height) 
{
    // Get the row and column of the pixel
    int col = get_global_id(0);
    int row = get_global_id(1);

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
            int neighbor_row = row + i;
            int neighbor_col = col + j;

            // Check for out-of-bound coordinates
            if (neighbor_row >= 0 && neighbor_row < height && 
                neighbor_col >= 0 && neighbor_col < width) {
                window[count++] = input_image[neighbor_row * width + neighbor_col];
            } else {
                // For out-of-bound pixels, add a default value (e.g., 0)
                window[count++] = 0;
            }
        }
    }

    // Sort the values in the window array
    for (int i = 0; i < count - 1; i++) {
        for (int j = i + 1; j < count; j++) {
            if (window[i] > window[j]) {
                uchar temp = window[i];
                window[i] = window[j];
                window[j] = temp;
            }
        }
    }

    // Set the output pixel value to the median of the window
    output_image[row * width + col] = window[count / 2];
}
