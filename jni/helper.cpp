#include <CL/cl2.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include <vector>

cl_platform_id platform;
cl_device_id device;
cl_context context;

#define CHECK_ERR(err, name)                                               \
    if (err != CL_SUCCESS)                                                 \
    {                                                                      \
        std::cerr << "Error: " << name << " (" << err << ")" << std::endl; \
        exit(EXIT_FAILURE);                                                \
    }

// Helper function to read a file into a string
std::string read_file(const char *filename)
{
    std::ifstream ifs(filename);
    std::string content((std::istreambuf_iterator<char>(ifs)),
                        (std::istreambuf_iterator<char>()));
    return content;
}

void init()
{
    cl_int err;
    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    std::cout << platform << std::endl
              << device << std::endl
              << context << std::endl;
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

int add_bin()
{
    cv::Mat image1 = cv::imread("image_add1.png", cv::IMREAD_UNCHANGED);
    cv::Mat image2 = cv::imread("image_add2.png", cv::IMREAD_UNCHANGED);

    if (image1.empty() || image2.empty())
    {
        std::cerr << "Error loading images!" << std::endl;
        return -1;
    }

    // Ensure images are of same size and type
    if (image1.size() != image2.size() || image1.type() != image2.type() || image1.channels() != 4)
    {
        std::cerr << "Images must be of the same size and type, with 4 channels (RGBA)!" << std::endl;
        return -1;
    }

    // Initialize OpenCL
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
    cl::CommandQueue queue(context, device, 0, &err);
    CHECK_ERR(err, "CommandQueue");

    // Load and build kernel
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
    cl::Image2D outputImage2d(context, CL_MEM_WRITE_ONLY,
                              format, image1.cols, image1.rows, 0, nullptr, &err);
    CHECK_ERR(err, "OutputImage");
    // Set kernel arguments
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

    // Enqueue kernel execution
    cl::NDRange global(image1.cols, image1.rows);
    cl::Event event;
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, nullptr, &event);
    CHECK_ERR(err, "EnqueueNDRangeKernel");

    // Read back the output

    event.wait();
    cl_int status = event.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>();
    if (status != CL_COMPLETE)
    {
        std::cerr << "Error during kernel execution" << std::endl;
    }

    std::vector<unsigned char> outputImageData(image1.total() * image1.elemSize());
    std::array<size_t, 3> origin;
    std::array<size_t, 3> region;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;
    region[0] = image1.cols;
    region[1] = image1.rows;
    region[2] = 1;
    err = queue.enqueueReadImage(outputImage2d, CL_TRUE, origin, region, 0, 0, outputImageData.data());
    // CHECK_ERR(err, "Read output image");
    cv::Mat outputMat(image1.rows, image1.cols, CV_8UC4, outputImageData.data());
    cv::imwrite("output_image.png", outputMat);
}

int gaussian_blur() {
    cv::Mat image1 = cv::imread("image_add1.png", cv::IMREAD_UNCHANGED);

    if (image1.empty())
    {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

    // Initialize OpenCL
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
    cl::CommandQueue queue(context, device, 0, &err);
    CHECK_ERR(err, "CommandQueue");

    // Load and build kernel
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
    // Set kernel arguments
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

    // Enqueue kernel execution
    cl::NDRange global(image1.cols, image1.rows);
    cl::Event event;
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, nullptr, &event);
    CHECK_ERR(err, "EnqueueNDRangeKernel");

    // Read back the output

    event.wait();
    cl_int status = event.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>();
    if (status != CL_COMPLETE)
    {
        std::cerr << "Error during kernel execution" << std::endl;
    }

    std::vector<unsigned char> outputImageData(image1.total() * image1.elemSize());
    std::array<size_t, 3> origin;
    std::array<size_t, 3> region;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;
    region[0] = image1.cols;
    region[1] = image1.rows;
    region[2] = 1;
    err = queue.enqueueReadBuffer(outputImage2d, CL_TRUE, 0, bufferSize, outputImageData.data());
    // CHECK_ERR(err, "Read output image");
    cv::Mat outputMat(image1.rows, image1.cols, CV_8UC4, outputImageData.data());
    cv::imwrite("output_image.png", outputMat);
}

int multiply_images() { //MULTIPLY
    cv::Mat image1 = cv::imread("image_add1.png", cv::IMREAD_UNCHANGED);
    cv::Mat image2 = cv::imread("image_add2.png", cv::IMREAD_UNCHANGED);

    if (image1.empty() || image2.empty()) {
        std::cerr << "Error loading images!" << std::endl;
        return -1;
    }

    // Ensure images are of same size and type
    if (image1.size() != image2.size() || image1.type() != image2.type() || image1.channels() != 4) {
        std::cerr << "Images must be of the same size and type, with 4 channels (RGBA)!" << std::endl;
        return -1;
    }

    // Initialize OpenCL
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
    cl::CommandQueue queue(context, device, 0, &err);
    CHECK_ERR(err, "CommandQueue");

    // Load and build kernel
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

    cl::Kernel kernel(program, "multiply_images", &err);
    CHECK_ERR(err, "Kernel");

    size_t bufferSize = image1.cols * image1.rows * image1.elemSize();
    cl::Buffer imageBuffer1(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bufferSize, image1.data, &err);
    CHECK_ERR(err, "BufferImage1");
    cl::Buffer imageBuffer2(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bufferSize, image2.data, &err);
    CHECK_ERR(err, "BufferImage2");
    cl::Buffer outputImageBuffer(context, CL_MEM_WRITE_ONLY, bufferSize, nullptr, &err);
    CHECK_ERR(err, "OutputImageBuffer");

    // Set kernel arguments
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

    // Enqueue kernel execution
    cl::NDRange global(image1.cols, image1.rows);
    cl::Event event;
    err = queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, nullptr, &event);
    CHECK_ERR(err, "EnqueueNDRangeKernel");

    // Read back the output
    event.wait();
    cl_int status = event.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>();
    if (status != CL_COMPLETE) {
        std::cerr << "Error during kernel execution" << std::endl;
        return -1;
    }

    std::vector<unsigned char> outputImageData(image1.total() * image1.elemSize());
    err = queue.enqueueReadBuffer(outputImageBuffer, CL_TRUE, 0, bufferSize, outputImageData.data());
    CHECK_ERR(err, "Read output image");

    cv::Mat outputMat(image1.rows, image1.cols, CV_8UC4, outputImageData.data());
    cv::imwrite("output_image.png", outputMat);

    return 0;
}
