
__kernel void crop(__global const uchar4* inputImage, int inputWidth, int inputHeight, int topCrop, int bottomCrop, __global uchar4* outputImage) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    int outputHeight = inputHeight - topCrop - bottomCrop;

    if (x < inputWidth && y < outputHeight) {
        int inputY = y + topCrop;
        int inputIndex = inputY * inputWidth + x;
        int outputIndex = y * inputWidth + x;
        outputImage[outputIndex] = inputImage[inputIndex];
    }
}
/*
#define TOP_CROP 10    
#define BOTTOM_CROP 10  
#define LEFT_CROP 10   
#define RIGHT_CROP 10  

__kernel void crop(__global const uchar* inputImage, 
                   const int inputWidth, const int inputHeight,
                   __global uchar* outputImage) {

    int outputWidth = inputWidth - LEFT_CROP - RIGHT_CROP;
    int outputHeight = inputHeight - TOP_CROP - BOTTOM_CROP;

    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= outputWidth || y >= outputHeight) {
        return;
    }

    int inputX = x + LEFT_CROP;
    int inputY = y + TOP_CROP;

    int inputIndex = (inputY * inputWidth + inputX) * 4; // Assuming 4 channels (RGBA)
    int outputIndex = (y * outputWidth + x) * 4;

    for (int i = 0; i < 4; i++) {
        outputImage[outputIndex + i] = inputImage[inputIndex + i];
    }
}
*/