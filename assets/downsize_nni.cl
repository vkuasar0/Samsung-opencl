__kernel void downsize(__global uchar* inImg, __global uchar* outImg, int inWidth, int inHeight, int outWidth, int outHeight) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= outWidth || y >= outHeight) return;

    float scaleX = (float)inWidth / outWidth;
    float scaleY = (float)inHeight / outHeight;

    float inX = x * scaleX;
    float inY = y * scaleY;

    int x1 = (int)inX;
    int y1 = (int)inY;
    int x2 = min(x1 + 1, inWidth - 1);
    int y2 = min(y1 + 1, inHeight - 1);

    float dx = inX - x1;
    float dy = inY - y1;

    for (int c = 0; c < 3; ++c) {
        float topLeft = inImg[(y1 * inWidth + x1) * 3 + c];
        float topRight = inImg[(y1 * inWidth + x2) * 3 + c];
        float bottomLeft = inImg[(y2 * inWidth + x1) * 3 + c];
        float bottomRight = inImg[(y2 * inWidth + x2) * 3 + c];

        float top = topLeft + dx * (topRight - topLeft);
        float bottom = bottomLeft + dx * (bottomRight - bottomLeft);
        float pixel = top + dy * (bottom - top);

        outImg[(y * outWidth + x) * 3 + c] = (uchar)clamp(pixel, 0.0f, 255.0f);
    }
}
