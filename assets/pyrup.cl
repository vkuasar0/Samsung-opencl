#define size 5
#define BORDER_REFLECT 1

__kernel void pyrup(__global uchar *src, __global uchar *dst, int width, int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height)
        return;
    
    int dst_index = x * width * 3 + y * 3;

    float kenel[size][size] = {
        {0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625},
        {0.015625, 0.0625, 0.09375, 0.0625, 0.015625},
        {0.0234375, 0.09375, 0.140625, 0.09375, 0.0234375},
        {0.015625, 0.0625, 0.09375, 0.0625, 0.015625},
        {0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625}};

    float r = 0.0, g = 0.0, b = 0.0;
    int px, py;

    #pragma unroll
    for (int kx = -2; kx <= 2; kx++) {
        px = x + kx;
    #if BORDER_REFLECT
        if (px < 0)
            px = -px;
        else if (px >= height)
            px = (2 * height) - px - 2;
    #else
        if (px < 0 || px >= height)
            continue;
    #endif

        for (int ky = -2; ky <= 2; ky++) {
            py = y + ky;
    #if BORDER_REFLECT
            if (py < 0)
                py = -py;
            else if (py >= width)
                py = (2 * width) - py - 2;
    #else
            if (py < 0 || py >= width)
                continue;
    #endif

            float w = kenel[kx + 2][ky + 2]*4;

            int index = px * width * 3 + py * 3;
            r += w * (float)src[index++];
            g += w * (float)src[index++];
            b += w * (float)src[index];
        }
    }
    dst[dst_index++] = (uchar)clamp((int)round(r), 0, 255);
    dst[dst_index++] = (uchar)clamp((int)round(g), 0, 255);
    dst[dst_index] = (uchar)clamp((int)round(b), 0, 255);
}