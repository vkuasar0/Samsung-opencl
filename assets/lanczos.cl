__kernel void lanczos(
    __read_only image2d_t inputImage1, 
    __read_only image2d_t inputImage, 
    __write_only image2d_t resultImage, 
    int inputRows, 
    int inputCols,
    int outputRows,
    int outputCols)
{
    const int2 outputCoords = (int2)(get_global_id(0), get_global_id(1));

    if (outputCoords.x >= outputCols || outputCoords.y >= outputRows)
        return;

    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

    // Use a scaleFactor to resize the width by half and keep the height the same
    float2 scaleFactor = (float2)(0.75f, 1.25f);

    // Calculate the corresponding coordinates in the input image
    float2 inputCoords = convert_float2(outputCoords) * scaleFactor;

    // Use explicit conversion functions to convert between float2 and int2
    int2 centerCoordinate = convert_int2(round(inputCoords));
    int2 oneStepLeftCoordinate = centerCoordinate + (int2)(-1, 0);
    int2 twoStepsLeftCoordinate = centerCoordinate + (int2)(-2, 0);
    int2 threeStepsLeftCoordinate = centerCoordinate + (int2)(-3, 0);
    int2 fourStepsLeftCoordinate = centerCoordinate + (int2)(-4, 0);
    int2 oneStepRightCoordinate = centerCoordinate + (int2)(1, 0);
    int2 twoStepsRightCoordinate = centerCoordinate + (int2)(2, 0);
    int2 threeStepsRightCoordinate = centerCoordinate + (int2)(3, 0);
    int2 fourStepsRightCoordinate = centerCoordinate + (int2)(4, 0);

    // Ensure coordinates are within bounds
    if (centerCoordinate.x < 0 || centerCoordinate.x >= inputCols ||
        centerCoordinate.y < 0 || centerCoordinate.y >= inputRows) {
        return;
    }

    float4 rgb = read_imagef(inputImage, sampler, centerCoordinate) * 0.38026f;

    if (oneStepLeftCoordinate.x >= 0 && oneStepLeftCoordinate.x < inputCols)
        rgb += read_imagef(inputImage, sampler, oneStepLeftCoordinate) * 0.27667f;
    if (oneStepRightCoordinate.x >= 0 && oneStepRightCoordinate.x < inputCols)
        rgb += read_imagef(inputImage, sampler, oneStepRightCoordinate) * 0.27667f;

    if (twoStepsLeftCoordinate.x >= 0 && twoStepsLeftCoordinate.x < inputCols)
        rgb += read_imagef(inputImage, sampler, twoStepsLeftCoordinate) * 0.08074f;
    if (twoStepsRightCoordinate.x >= 0 && twoStepsRightCoordinate.x < inputCols)
        rgb += read_imagef(inputImage, sampler, twoStepsRightCoordinate) * 0.08074f;

    if (threeStepsLeftCoordinate.x >= 0 && threeStepsLeftCoordinate.x < inputCols)
        rgb += read_imagef(inputImage, sampler, threeStepsLeftCoordinate) * -0.02612f;
    if (threeStepsRightCoordinate.x >= 0 && threeStepsRightCoordinate.x < inputCols)
        rgb += read_imagef(inputImage, sampler, threeStepsRightCoordinate) * -0.02612f;

    if (fourStepsLeftCoordinate.x >= 0 && fourStepsLeftCoordinate.x < inputCols)
        rgb += read_imagef(inputImage, sampler, fourStepsLeftCoordinate) * -0.02143f;
    if (fourStepsRightCoordinate.x >= 0 && fourStepsRightCoordinate.x < inputCols)
        rgb += read_imagef(inputImage, sampler, fourStepsRightCoordinate) * -0.02143f;

    // Write the computed color to the output image
    write_imagef(resultImage, outputCoords, rgb);
}
