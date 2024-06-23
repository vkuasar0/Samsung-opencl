/*
  __kernel void downsize(__global uchar* input, __global uchar* output, int width, int height, int new_width, int new_height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);  
    // Calculate the scaling factor for x and y
    float sx = (float)width / new_width;
    float sy = (float)height / new_height;  
    // Calculate the position of the current pixel in the input image
    float ix = (float)x * sx;
    float iy = (float)y * sy;   
    // Calculate the integer and fractional parts of the position
    int ix0 = (int)floor(ix);
    int iy0 = (int)floor(iy);
    float dx = ix - ix0;
    float dy = iy - iy0;
    // Calculate the coefficients for the bicubic interpolation
    float c[4];
    c[0] = ((-0.5 * dx + 1.0) * dx - 0.5) * dx;
    c[1] = ((1.5 * dx - 2.5) * dx) * dx + 1.0;
    c[2] = ((-1.5 * dx + 2.0) * dx + 0.5) * dx;
    c[3] = ((0.5 * dx - 0.5) * dx) * dx;
    float r[4];
    r[0] = ((-0.5 * dy + 1.0) * dy - 0.5) * dy;
    r[1] = ((1.5 * dy - 2.5) * dy) * dy + 1.0;
    r[2] = ((-1.5 * dy + 2.0) * dy + 0.5) * dy;
    r[3] = ((0.5 * dy - 0.5) * dy) * dy; 
    // Interpolate each color channel separately
    for (int k = 0; k < 3; k++)
    {
        float sum = 0.0;
        for (int i = -1; i <= 2; i++)
        {
            int px = clamp(ix0 + i, 0, width - 1);
            float wx = c[i + 1];
            for (int j = -1; j <= 2; j++)
            {
                int py = clamp(iy0 + j, 0, height - 1);
                float wy = r[j + 1];
                float val = (float)input[(py * width + px) * 3 + k];
                sum += wx * wy * val;
            }
        }
        output[(y * new_width + x) * 3 + k] = (uchar)clamp(sum, 0.0, 255.0);
    }
}*/


int reflect_border_index(int index, int size) {
  if (index < 0)
    return 0;
  if (index >= size)
    return size-1;
  return index;
}

__kernel void downsize(__global uchar *input, __global uchar *output, int width,
                       int height, int new_width, int new_height) {
  int2 gid = (int2)(get_global_id(0), get_global_id(1));

  // Calculate the scaling factor for x and y
  float2 scale = (float2)((float)(width) / (float)(new_width),
                          (float)(height) / (float)(new_height));

  float2 pos = (float2)((float)((gid.x + 0.5) * scale.x - 0.5),
                        (float)((gid.y + 0.5) * scale.y - 0.5));

  // Calculate the integer and fractional parts of the position
  int2 pos_int = (int2)(floor(pos.x), floor(pos.y));
  float2 pos_frac =
      (float2)(pos.x - (float)pos_int.x, pos.y - (float)pos_int.y);

  // Calculate the coefficients for the bicubic interpolation
  float4 c, r;

  const float A = -0.75f;


  c.x = ((((A * (pos_frac.x + 1.f) - 5 * A) * (pos_frac.x + 1.f) + 8 * A) *
              (pos_frac.x + 1.f) -
          4 * A));
  c.y = (((A + 2) * pos_frac.x - (A + 3)) * pos_frac.x * pos_frac.x + 1.f);
  c.z = ((((A + 2) * (1.f - pos_frac.x) - (A + 3)) * (1.f - pos_frac.x) *
              (1.f - pos_frac.x) +
          1.f));
  c.w = ((1.f - c.x - c.y - c.z));

  r.x = ((((A * (pos_frac.y + 1.f) - 5 * A) * (pos_frac.y + 1.f) + 8 * A) *
              (pos_frac.y + 1.f) -
          4 * A));
  r.y =
      ((0.f, ((A + 2) * pos_frac.y - (A + 3)) * pos_frac.y * pos_frac.y + 1.f));
  r.z = ((((A + 2) * (1.f - pos_frac.y) - (A + 3)) * (1.f - pos_frac.y) *
              (1.f - pos_frac.y) +
          1.f));
  r.w = ((1.f - r.x - r.y - r.z));



  for (int k = 0; k < 3; k++) {
    float4 sum = (float4)(0.0f);

    int4 py = (int4)(pos_int.y - 1, pos_int.y, pos_int.y + 1, pos_int.y + 2);
    int4 px = (int4)(pos_int.x - 1, pos_int.x, pos_int.x + 1, pos_int.x + 2);

    py.x = reflect_border_index(py.x, height);
    py.y = reflect_border_index(py.y, height);
    py.z = reflect_border_index(py.z, height);
    py.w = reflect_border_index(py.w, height);

    px.x = reflect_border_index(px.x, width);
    px.y = reflect_border_index(px.y, width);
    px.z = reflect_border_index(px.z, width);
    px.w = reflect_border_index(px.w, width);

    // Clamp the range of pixels to avoid accessing out-of-bounds pixels
    // py = clamp(py, 0, height - 1);
    // px = clamp(px, 0, width - 1);

    float4 val = (float4)((float)input[(py.x * width + px.x) * 3 + k],
                          (float)input[(py.x * width + px.y) * 3 + k],
                          (float)input[(py.x * width + px.z) * 3 + k],
                          (float)input[(py.x * width + px.w) * 3 + k]);

    sum += c * val * r.x;

    val = (float4)((float)input[(py.y * width + px.x) * 3 + k],
                   (float)input[(py.y * width + px.y) * 3 + k],
                   (float)input[(py.y * width + px.z) * 3 + k],
                   (float)input[(py.y * width + px.w) * 3 + k]);
    sum += c * val * r.y;

    val = (float4)((float)input[(py.z * width + px.x) * 3 + k],
                   (float)input[(py.z * width + px.y) * 3 + k],
                   (float)input[(py.z * width + px.z) * 3 + k],
                   (float)input[(py.z * width + px.w) * 3 + k]);
    sum += c * val * r.z;

    val = (float4)((float)input[(py.w * width + px.x) * 3 + k],
                   (float)input[(py.w * width + px.y) * 3 + k],
                   (float)input[(py.w * width + px.z) * 3 + k],
                   (float)input[(py.w * width + px.w) * 3 + k]);
    sum += c * val * r.w;

    output[(gid.y * new_width + gid.x) * 3 + k] =
        (uchar)((int)clamp(((sum.x + sum.y + sum.z + sum.w+0.5)), 0.f, 255.f));
  }
}
