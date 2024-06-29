__kernel void emboss(__read_only image2d_t inputImage, 
                     __write_only image2d_t resultImage, 
                     sampler_t sampler)
{
    int2 coords = (int2)(get_global_id(0), get_global_id(1));

    float avg[9];
    int n = 0;

    for (int i = -1; i <= 1; ++i)
    {
        for (int j = -1; j <= 1; ++j)
        {
            int2 offset_coords = coords + (int2)(i, j);
            float4 pixel = read_imagef(inputImage, sampler, offset_coords);
            avg[n++] = (pixel.x + pixel.y + pixel.z) / 3.0f;
        }
    }

    float k[9];
    k[0] = -1.0f; 
    k[1] = 0.0f; 
    k[2] = 0.0f;
    k[3] = 0.0f;  
    k[4] = -1.0f; 
    k[5] = 0.0f;
    k[6] = 0.0f;  
    k[7] = 0.0f;  
    k[8] = 2.0f;

    float res = 0.0f;
    for (int i = 0; i < 9; ++i)
    {
        res += k[i] * avg[i];
    }
    res = clamp(res + 0.5f, 0.0f, 1.0f);

    float4 resultColor = (float4)(res, res, res, 1.0f);
    write_imagef(resultImage, coords, resultColor);
}
