#include "helper.h"

cl_platform_id platform;
cl_device_id device;
cl_context context;

#define CHECK_ERR(err, name)                                               \
    if (err != CL_SUCCESS)                                                 \
    {                                                                      \
        std::cerr << "Error: " << name << " (" << err << ")" << std::endl; \
        exit(EXIT_FAILURE);                                                \
    }

std::string loadKernel(const char *filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Failed to open kernel file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    return std::string((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
}

double add_bin()
{
    cv::Mat image1 = cv::imread("image_add1.png", cv::IMREAD_UNCHANGED);
    cv::Mat image2 = cv::imread("image_add2.png", cv::IMREAD_UNCHANGED);

    if (image1.empty() || image2.empty())
    {
        std::cerr << "Error loading images!" << std::endl;
        return -1;
    }

    if (image1.size() != image2.size() || image1.type() != image2.type() || image1.channels() != 4)
    {
        std::cerr << "Images must be of the same size and type, with 4 channels (RGBA)!" << std::endl;
        return -1;
    }

    cl_int err;
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty())
    {
        std::cerr << "No OpenCL platforms found!" << std::endl;
        return -1;
    }

    cl::Platform platform = platforms.front();
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty())
    {
        std::cerr << "No OpenCL devices found!" << std::endl;
        return -1;
    }

    cl::Device device = devices.front();
    cl::Context context(device);
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_ERR(err, "CommandQueue");

    std::string kernelSource = loadKernel("add.cl");
    cl::Program::Sources sources;
    sources.push_back({kernelSource.c_str(), kernelSource.length()});

    cl::Program program(context, sources);
    err = program.build({device});
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error building kernel: " << err << std::endl;
        std::cerr << "Build log: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        return -1;
    }

    cl::Kernel kernel(program, "add_images", &err);
    CHECK_ERR(err, "Kernel");

    cl::ImageFormat format(CL_RGBA, CL_UNSIGNED_INT8);
    cl::Image2D image2d1(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         format, image1.cols, image1.rows, 0, image1.data, &err);
    CHECK_ERR(err, "BufferImage1");
    cl::Image2D image2d2(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         format, image2.cols, image2.rows, 0, image2.data, &err);
    CHECK_ERR(err, "BufferImage2");
    cl::Image2D outputImage2d(context, CL_MEM_READ_WRITE,
                              format, image1.cols, image1.rows, 0, nullptr, &err);
    CHECK_ERR(err, "OutputImage");
    err = kernel.setArg(0, image2d1);
    CHECK_ERR(err, "SetArg 0");
    err = kernel.setArg(1, image2d2);
    CHECK_ERR(err, "SetArg 1");
    err = kernel.setArg(2, outputImage2d);
    CHECK_ERR(err, "SetArg 2");
    err = kernel.setArg(3, image1.cols);
    CHECK_ERR(err, "SetArg 3");
    err = kernel.setArg(4, image1.rows);
    CHECK_ERR(err, "SetArg 4");

    cl::NDRange global(image1.cols, image1.rows);
    cl::Event event;
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, nullptr, &event);
    CHECK_ERR(err, "EnqueueNDRangeKernel");

    event.wait();

    cl_ulong start_time, end_time;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_time);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_time);

    cl_int status = event.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>();
    if (status != CL_COMPLETE)
    {
        std::cerr << "Error during kernel execution" << std::endl;
    }

    std::vector<unsigned char> outputImageData(image1.total() * image1.elemSize());
    std::array<size_t, 3> origin = {0, 0, 0};
    std::array<size_t, 3> region = {static_cast<unsigned long>(image1.cols), static_cast<unsigned long>(image1.rows), 1};
    err = queue.enqueueReadImage(outputImage2d, CL_TRUE, origin, region, 0, 0, outputImageData.data());
    CHECK_ERR(err, "Read output image");
    queue.finish();
    cv::Mat outputMat(image1.rows, image1.cols, CV_8UC4, outputImageData.data());
    cv::imwrite("output_add.png", outputMat);
    return (end_time - start_time)/1000.0;
}

double gaussian_blur() {
    cv::Mat image1 = cv::imread("image_add1.png", cv::IMREAD_UNCHANGED);

    if (image1.empty())
    {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    cl_int err;
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty())
    {
        std::cerr << "No OpenCL platforms found!" << std::endl;
        return -1;
    }

    cl::Platform platform = platforms.front();
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty())
    {
        std::cerr << "No OpenCL devices found!" << std::endl;
        return -1;
    }

    cl::Device device = devices.front();
    cl::Context context(device);
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_ERR(err, "CommandQueue");

    std::string kernelSource = loadKernel("gaussianBlur.cl");
    cl::Program::Sources sources;
    sources.push_back({kernelSource.c_str(), kernelSource.length()});

    cl::Program program(context, sources);
    err = program.build({device});
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error building kernel: " << err << std::endl;
        std::cerr << "Build log: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        return -1;
    }

    cl::Kernel kernel(program, "gaussian", &err);
    CHECK_ERR(err, "Kernel");
    size_t bufferSize = image1.cols*image1.rows*image1.elemSize();
    cl::Buffer image2d1(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bufferSize, image1.data, &err);
    CHECK_ERR(err, "BufferImage1");
    cl::Buffer outputImage2d(context, CL_MEM_WRITE_ONLY, bufferSize, nullptr, &err);
    CHECK_ERR(err, "OutputImage");
    err = kernel.setArg(0, image2d1);
    CHECK_ERR(err, "SetArg 0");
    err = kernel.setArg(1, outputImage2d);
    CHECK_ERR(err, "SetArg 1");
    err = kernel.setArg(2, image1.rows);
    CHECK_ERR(err, "SetArg 2");
    err = kernel.setArg(3, image1.cols);
    CHECK_ERR(err, "SetArg 3");
    err = kernel.setArg(4, 30);
    CHECK_ERR(err, "SetArg 4");
    err = kernel.setArg(5, 50.0f);
    CHECK_ERR(err, "SetArg 5");

    cl::NDRange global(image1.cols, image1.rows);
    cl::Event event;
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, nullptr, &event);
    CHECK_ERR(err, "EnqueueNDRangeKernel");
    event.wait();

    cl_ulong start_time, end_time;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_time);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_time);

    cl_int status = event.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>();
    if (status != CL_COMPLETE)
    {
        std::cerr << "Error during kernel execution" << std::endl;
    }

    std::vector<unsigned char> outputImageData(image1.total() * image1.elemSize());
    std::array<size_t, 3> origin = {0, 0, 0};
    std::array<size_t, 3> region = {static_cast<unsigned long>(image1.cols), static_cast<unsigned long>(image1.rows), 1};
    err = queue.enqueueReadBuffer(outputImage2d, CL_TRUE, 0, bufferSize, outputImageData.data());
    CHECK_ERR(err, "Read output image");
    queue.finish();
    cv::Mat outputMat(image1.rows, image1.cols, CV_8UC4, outputImageData.data());
    cv::imwrite("output_gaussian.png", outputMat);
    return (end_time - start_time)/1000.0;
}

double multiply_images() {  //MULTIPLY
    cv::Mat image1 = cv::imread("image_add1.png", cv::IMREAD_UNCHANGED);
    cv::Mat image2 = cv::imread("image_add2.png", cv::IMREAD_UNCHANGED);

    if (image1.empty() || image2.empty()) {
        std::cerr << "Error loading images!" << std::endl;
        return -1;
    }

    if (image1.size() != image2.size() || image1.type() != image2.type() || image1.channels() != 4) {
        std::cerr << "Images must be of the same size and type, with 4 channels (RGBA)!" << std::endl;
        return -1;
    }

    cl_int err;
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        std::cerr << "No OpenCL platforms found!" << std::endl;
        return -1;
    }

    cl::Platform platform = platforms.front();
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty()) {
        std::cerr << "No OpenCL devices found!" << std::endl;
        return -1;
    }

    cl::Device device = devices.front();
    cl::Context context(device);
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_ERR(err, "CommandQueue");

    std::string kernelSource = loadKernel("mul.cl");
    cl::Program::Sources sources;
    sources.push_back({kernelSource.c_str(), kernelSource.length()});
    cl::Program program(context, sources);
    err = program.build({device});
    if (err != CL_SUCCESS) {
        std::cerr << "Error building kernel: " << err << std::endl;
        std::cerr << "Build log: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        return -1;
    }

    cl::Kernel kernel(program, "mul", &err);
    CHECK_ERR(err, "Kernel");

    size_t bufferSize = image1.cols * image1.rows * image1.elemSize();
    cl::Buffer imageBuffer1(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bufferSize, image1.data, &err);
    CHECK_ERR(err, "BufferImage1");
    cl::Buffer imageBuffer2(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bufferSize, image2.data, &err);
    CHECK_ERR(err, "BufferImage2");
    cl::Buffer outputImageBuffer(context, CL_MEM_WRITE_ONLY, bufferSize, nullptr, &err);
    CHECK_ERR(err, "OutputImageBuffer");

    err = kernel.setArg(0, imageBuffer1);
    CHECK_ERR(err, "SetArg 0");
    err = kernel.setArg(1, imageBuffer2);
    CHECK_ERR(err, "SetArg 1");
    err = kernel.setArg(2, outputImageBuffer);
    CHECK_ERR(err, "SetArg 2");
    err = kernel.setArg(3, image1.cols);
    CHECK_ERR(err, "SetArg 3");
    err = kernel.setArg(4, image1.rows);
    CHECK_ERR(err, "SetArg 4");

    cl::NDRange global(image1.cols, image1.rows);
    cl::Event event;
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, nullptr, &event);
    CHECK_ERR(err, "EnqueueNDRangeKernel");

    event.wait();
    cl_ulong start_time, end_time;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_time);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_time);

    cl_int status = event.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>();
    if (status != CL_COMPLETE) {
        std::cerr << "Error during kernel execution" << std::endl;
        return -1;
    }

    std::vector<unsigned char> outputImageData(image1.total() * image1.elemSize());
    err = queue.enqueueReadBuffer(outputImageBuffer, CL_TRUE, 0, bufferSize, outputImageData.data());
    CHECK_ERR(err, "Read output image");
    queue.finish();

    cv::Mat outputMat(image1.rows, image1.cols, CV_8UC4, outputImageData.data());
    cv::imwrite("output_multiply.png", outputMat);

    return (end_time - start_time) / 1000.0;
}


double crop() {
    cv::Mat image1 = cv::imread("image_add1.png", cv::IMREAD_UNCHANGED);

    if (image1.empty()) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    cl_int err;
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        std::cerr << "No OpenCL platforms found!" << std::endl;
        return -1;
    }

    cl::Platform platform = platforms.front();
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty()) {
        std::cerr << "No OpenCL devices found!" << std::endl;
        return -1;
    }

    cl::Device device = devices.front();
    cl::Context context(device);
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_ERR(err, "CommandQueue");

    std::string kernelSource = loadKernel("crop.cl");
    cl::Program::Sources sources;
    sources.push_back({kernelSource.c_str(), kernelSource.length()});

    cl::Program program(context, sources);
    err = program.build({device});
    if (err != CL_SUCCESS) {
        std::cerr << "Error building kernel: " << err << std::endl;
        std::cerr << "Build log: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        return -1;
    }

    cl::Kernel kernel(program, "crop", &err);
    CHECK_ERR(err, "Kernel");

    size_t bufferSize = image1.cols * image1.rows * image1.elemSize();
    cl::Buffer imageBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bufferSize, image1.data, &err);
    CHECK_ERR(err, "BufferImage1");

    // Calculate the dimensions of the output image
    int top_crop = static_cast<int>(0.2f * image1.rows); // 20% of the image height
    int bottom_crop = static_cast<int>(0.1f * image1.rows); // 10% of the image height
    int output_height = image1.rows - top_crop - bottom_crop;
    int output_width = image1.cols;

    size_t outputBufferSize = output_width * output_height * image1.elemSize();
    cl::Buffer outputBuffer(context, CL_MEM_WRITE_ONLY, outputBufferSize, nullptr, &err);
    CHECK_ERR(err, "OutputImage");

    err = kernel.setArg(0, imageBuffer);
    CHECK_ERR(err, "SetArg 0");
    err = kernel.setArg(1, image1.cols);
    CHECK_ERR(err, "SetArg 1");
    err = kernel.setArg(2, image1.rows);
    CHECK_ERR(err, "SetArg 2");
    err = kernel.setArg(3, top_crop);
    CHECK_ERR(err, "SetArg 3");
    err = kernel.setArg(4, bottom_crop);
    CHECK_ERR(err, "SetArg 4");
    err = kernel.setArg(5, outputBuffer);
    CHECK_ERR(err, "SetArg 5");

    cl::NDRange global(output_width, output_height);
    cl::Event event;
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, nullptr, &event);
    CHECK_ERR(err, "EnqueueNDRangeKernel");
    event.wait();

    cl_ulong start_time, end_time;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_time);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_time);

    cl_int status = event.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>();
    if (status != CL_COMPLETE) {
        std::cerr << "Error during kernel execution" << std::endl;
    }

    std::vector<unsigned char> outputImageData(outputBufferSize);
    err = queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, outputBufferSize, outputImageData.data());
    CHECK_ERR(err, "Read output image");
    queue.finish();

    cv::Mat outputMat(output_height, output_width, image1.type(), outputImageData.data());
    cv::imwrite("output_crop.png", outputMat);

    return (end_time - start_time) / 1000.0;
}
/*
double crop() {
    cv::Mat image1 = cv::imread("image_add1.png", cv::IMREAD_UNCHANGED);

    if (image1.empty()) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    cl_int err;
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        std::cerr << "No OpenCL platforms found!" << std::endl;
        return -1;
    }

    cl::Platform platform = platforms.front();
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty()) {
        std::cerr << "No OpenCL devices found!" << std::endl;
        return -1;
    }

    cl::Device device = devices.front();
    cl::Context context(device);
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_ERR(err, "CommandQueue");

    std::string kernelSource = loadKernel("crop.cl");
    cl::Program::Sources sources;
    sources.push_back({kernelSource.c_str(), kernelSource.length()});

    cl::Program program(context, sources);
    err = program.build({device});
    if (err != CL_SUCCESS) {
        std::cerr << "Error building kernel: " << err << std::endl;
        std::cerr << "Build log: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        return -1;
    }

    cl::Kernel kernel(program, "crop", &err);
    CHECK_ERR(err, "Kernel");

    size_t bufferSize = image1.cols * image1.rows * image1.elemSize();
    cl::Buffer imageBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bufferSize, image1.data, &err);
    CHECK_ERR(err, "BufferImage1");

    // Calculate the dimensions of the output image
    const int top_crop = 10;
    const int bottom_crop = 10;
    const int left_crop = 10;
    const int right_crop = 10;
    int output_height = image1.rows - top_crop - bottom_crop;
    int output_width = image1.cols - left_crop - right_crop;

    size_t outputBufferSize = output_width * output_height * image1.elemSize();
    cl::Buffer outputBuffer(context, CL_MEM_WRITE_ONLY, outputBufferSize, nullptr, &err);
    CHECK_ERR(err, "OutputImage");

    err = kernel.setArg(0, imageBuffer);
    CHECK_ERR(err, "SetArg 0");
    err = kernel.setArg(1, image1.cols);
    CHECK_ERR(err, "SetArg 1");
    err = kernel.setArg(2, image1.rows);
    CHECK_ERR(err, "SetArg 2");
    err = kernel.setArg(3, outputBuffer);
    CHECK_ERR(err, "SetArg 3");

    cl::NDRange global(output_width, output_height);
    cl::Event event;
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, nullptr, &event);
    CHECK_ERR(err, "EnqueueNDRangeKernel");
    event.wait();

    cl_ulong start_time, end_time;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_time);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_time);

    cl_int status = event.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>();
    if (status != CL_COMPLETE) {
        std::cerr << "Error during kernel execution" << std::endl;
    }

    std::vector<unsigned char> outputImageData(outputBufferSize);
    err = queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, outputBufferSize, outputImageData.data());
    CHECK_ERR(err, "Read output image");
    queue.finish();

    cv::Mat outputMat(output_height, output_width, image1.type(), outputImageData.data());
    cv::imwrite("output_crop.png", outputMat);

    return (end_time - start_time) / 1000.0;
}*/

double lanczos() {
    cv::Mat image1 = cv::imread("image_add1.png", cv::IMREAD_UNCHANGED);

    if (image1.empty()) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    cl_int err;
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        std::cerr << "No OpenCL platforms found!" << std::endl;
        return -1;
    }

    cl::Platform platform = platforms.front();
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty()) {
        std::cerr << "No OpenCL devices found!" << std::endl;
        return -1;
    }

    cl::Device device = devices.front();
    cl::Context context(device);
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_ERR(err, "CommandQueue");

    std::string kernelSource = loadKernel("lanczos.cl");
    cl::Program::Sources sources;
    sources.push_back({kernelSource.c_str(), kernelSource.length()});

    cl::Program program(context, sources);
    err = program.build({device});
    if (err != CL_SUCCESS) {
        std::cerr << "Error building kernel: " << err << std::endl;
        std::cerr << "Build log: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        return -1;
    }

    cl::Kernel kernel(program, "lanczos", &err);
    CHECK_ERR(err, "Kernel");

    cl::ImageFormat format(CL_RGBA, CL_UNORM_INT8);

    cl::Image2D image2d1(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, format, image1.cols, image1.rows, 0, image1.data, &err);
    CHECK_ERR(err, "Image2D inputImage1");

    cl::Image2D inputImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, format, image1.cols, image1.rows, 0, image1.data, &err);
    CHECK_ERR(err, "Image2D inputImage");

    // Determine the output size (for this example, we will use half the size)
    int outputCols = static_cast<int>(image1.cols / 0.75f);
    int outputRows = static_cast<int>(image1.rows / 1.25f);


    cl::Image2D resultImage(context, CL_MEM_WRITE_ONLY, format, outputCols, outputRows, 0, nullptr, &err);
    CHECK_ERR(err, "Image2D resultImage");

    err = kernel.setArg(0, image2d1);
    CHECK_ERR(err, "SetArg 0");
    err = kernel.setArg(1, inputImage);
    CHECK_ERR(err, "SetArg 1");
    err = kernel.setArg(2, resultImage);
    CHECK_ERR(err, "SetArg 2");
    err = kernel.setArg(3, image1.rows);
    CHECK_ERR(err, "SetArg 3");
    err = kernel.setArg(4, image1.cols);
    CHECK_ERR(err, "SetArg 4");
    err = kernel.setArg(5, outputRows);
    CHECK_ERR(err, "SetArg 5");
    err = kernel.setArg(6, outputCols);
    CHECK_ERR(err, "SetArg 6");

    cl::NDRange global(outputCols, outputRows);
    cl::Event event;
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, nullptr, &event);
    CHECK_ERR(err, "EnqueueNDRangeKernel");
    event.wait();

    cl_ulong start_time, end_time;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_time);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_time);

    cl_int status = event.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>();
    if (status != CL_COMPLETE) {
        std::cerr << "Error during kernel execution" << std::endl;
    }

    size_t bufferSize = outputCols * outputRows * 4;
    std::vector<unsigned char> outputImageData(bufferSize);

    std::array<size_t, 3> origin = {0, 0, 0};
    std::array<size_t, 3> region = {static_cast<size_t>(outputCols), static_cast<size_t>(outputRows), 1};

    err = queue.enqueueReadImage(resultImage, CL_TRUE, origin, region, 0, 0, outputImageData.data());
    CHECK_ERR(err, "Read output image");
    queue.finish();

    cv::Mat outputMat(outputRows, outputCols, CV_8UC4, outputImageData.data());
    cv::imwrite("output_lanczos.png", outputMat);

    return (end_time - start_time) / 1000.0;
}

double emboss()
{
    cv::Mat image1 = cv::imread("image_add1.png", cv::IMREAD_UNCHANGED);

    if (image1.empty())
    {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    cl_int err;
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty())
    {
        std::cerr << "No OpenCL platforms found!" << std::endl;
        return -1;
    }

    cl::Platform platform = platforms.front();
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty())
    {
        std::cerr << "No OpenCL devices found!" << std::endl;
        return -1;
    }

    cl::Device device = devices.front();
    cl::Context context(device);
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_ERR(err, "CommandQueue");

    std::string kernelSource = loadKernel("emboss.cl");
    cl::Program::Sources sources;
    sources.push_back({kernelSource.c_str(), kernelSource.length()});

    cl::Program program(context, sources);
    err = program.build({device});
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error building kernel: " << err << std::endl;
        std::cerr << "Build log: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        return -1;
    }

    cl::Kernel kernel(program, "emboss", &err);
    CHECK_ERR(err, "Kernel");

    cl::ImageFormat format(CL_RGBA, CL_UNORM_INT8);
    cl::Image2D image2d1(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, format, image1.cols, image1.rows, 0, image1.data, &err);
    CHECK_ERR(err, "Image2D inputImage1");
    cl::Image2D outputImage2d(context, CL_MEM_WRITE_ONLY, format, image1.cols, image1.rows, 0, nullptr, &err);
    CHECK_ERR(err, "Image2D outputImage");

    cl::Sampler sampler(context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &err);
    CHECK_ERR(err, "Sampler");

    err = kernel.setArg(0, image2d1);
    CHECK_ERR(err, "SetArg 0");
    err = kernel.setArg(1, outputImage2d);
    CHECK_ERR(err, "SetArg 1");
    err = kernel.setArg(2, sampler);
    CHECK_ERR(err, "SetArg 2");

    cl::NDRange global(image1.cols, image1.rows);
    cl::Event event;
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, nullptr, &event);
    CHECK_ERR(err, "EnqueueNDRangeKernel");

    event.wait();

    cl_ulong start_time, end_time;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_time);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_time);

    cl_int status = event.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>();
    if (status != CL_COMPLETE)
    {
        std::cerr << "Error during kernel execution" << std::endl;
    }

    std::vector<unsigned char> outputImageData(image1.total() * image1.elemSize());
    std::array<size_t, 3> origin = {0, 0, 0};
    std::array<size_t, 3> region = {static_cast<unsigned long>(image1.cols), static_cast<unsigned long>(image1.rows), 1};
    err = queue.enqueueReadImage(outputImage2d, CL_TRUE, origin, region, 0, 0, outputImageData.data());
    CHECK_ERR(err, "Read output image");
    queue.finish();

    cv::Mat outputMat(image1.rows, image1.cols, CV_8UC4, outputImageData.data());
    cv::imwrite("output_emboss.png", outputMat);
    return (end_time - start_time) / 1000.0;
}

double gray_bgr() {
    // Load input image
    cv::Mat image1 = cv::imread("image_add1.png", cv::IMREAD_COLOR);

    if (image1.empty()) {
        std::cerr << "Error loading image!" << std::endl;
        return -1.0;
    }

    // Load kernel source
    std::string kernelSource = loadKernel("gray_bgr.cl");

    // Initialize OpenCL environment
    cl_int err;
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        std::cerr << "No OpenCL platforms found!" << std::endl;
        return -1.0;
    }

    cl::Platform platform = platforms.front();
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty()) {
        std::cerr << "No OpenCL devices found!" << std::endl;
        return -1.0;
    }

    cl::Device device = devices.front();
    cl::Context context(device);
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_ERR(err, "CommandQueue");

    // Build the OpenCL program
    cl::Program::Sources sources;
    sources.push_back({kernelSource.c_str(), kernelSource.length()});
    cl::Program program(context, sources);
    err = program.build({device});
    if (err != CL_SUCCESS) {
        std::cerr << "Error building kernel: " << err << std::endl;
        std::cerr << "Build log: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
        return -1.0;
    }

    // Create kernel
    cl::Kernel kernel(program, "gray_bgr", &err);
    CHECK_ERR(err, "Kernel");

    // Create input and output buffers
    size_t bufferSize = image1.cols * image1.rows * image1.channels() * sizeof(uchar);
    cl::Buffer inputImageBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bufferSize, image1.data, &err);
    CHECK_ERR(err, "InputImageBuffer");

    cl::Buffer outputImageBuffer(context, CL_MEM_WRITE_ONLY, image1.rows * image1.cols * sizeof(uchar), nullptr, &err);
    CHECK_ERR(err, "OutputImageBuffer");

    // Set kernel arguments
    err = kernel.setArg(0, inputImageBuffer);
    CHECK_ERR(err, "InputImage");
    err = kernel.setArg(1, outputImageBuffer);
    CHECK_ERR(err, "OutputImage");
    err = kernel.setArg(2, static_cast<int>(image1.cols));
    CHECK_ERR(err, "Width");
    err = kernel.setArg(3, static_cast<int>(image1.rows));
    CHECK_ERR(err, "Height");
    err = kernel.setArg(4, static_cast<int>(image1.channels()));
    CHECK_ERR(err, "Channels");

    // Enqueue kernel
    cl::NDRange global(image1.cols, image1.rows);
    cl::Event event;
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, nullptr, &event);
    CHECK_ERR(err, "EnqueueNDRangeKernel");

    // Wait for kernel to finish
    event.wait();

    // Timing information
    cl_ulong start_time, end_time;
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start_time);
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end_time);

    // Check execution status
    cl_int status = event.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>();
    if (status != CL_COMPLETE) {
        std::cerr << "Error during kernel execution" << std::endl;
        return -1.0;
    }

    // Read back the processed data
    std::vector<uchar> outputImageData(image1.rows * image1.cols);
    err = queue.enqueueReadBuffer(outputImageBuffer, CL_TRUE, 0, image1.rows * image1.cols * sizeof(uchar), outputImageData.data());
    CHECK_ERR(err, "Read output image");

    // Ensure all commands in the queue are finished
    queue.finish();

    // Create grayscale output image
    cv::Mat outputMat(image1.rows, image1.cols, CV_8UC1, outputImageData.data());
    cv::imwrite("output_gray_background.png", outputMat);

    // Calculate and return execution time in milliseconds
    double executionTime = static_cast<double>(end_time - start_time) / 1000.0;
    return executionTime;
}

