// #include <cuda.h>
#include <stdio.h>
#include <malloc.h>

typedef struct
{
    unsigned char R;
    unsigned char G;
    unsigned char B;
} Pixel;

typedef struct
{
    unsigned char V;
} GrayscalePixel;

const int BLUR_SIZE = 1;

void read_cat_image(Pixel *image, int height, int width);

void write_cat_image(GrayscalePixel *grayImage, int width, int height, const char filename[]);

__global__ void colorToGrayscaleKernel(GrayscalePixel *Pout, Pixel *Pin, int width, int height)
{
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    if (Col < width && Row < height)
    {
        Pixel *rgbPixel = &Pin[Row * width + Col];
        GrayscalePixel *grayPixel = &Pout[Row * width + Col];
        grayPixel->V = 0.21f * rgbPixel->R +  0.72f * rgbPixel->G +  0.07f * rgbPixel->B;
    }
}

__global__ void blurGrayscaleKernel(GrayscalePixel *imIn, GrayscalePixel *imOut, int width, int height){
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    if (Col < width && Row < height){
        int pixelVal=0, pixelCount=0;
        for(int blurCol = -BLUR_SIZE; blurCol<BLUR_SIZE+1; blurCol++){
            for(int blurRow = -BLUR_SIZE; blurRow<BLUR_SIZE+1; blurRow++){
                int inCol = Col + blurCol, inRow = Row + blurRow;
                if ((inCol >= 0) && (inCol < width) && (inRow >= 0) && (inRow < height)){
                    pixelVal += imIn[inRow * width + inCol].V;
                    pixelCount++;
                }
            }
        }
        imOut[Row*width + Col].V = (unsigned char) ((float) pixelVal / pixelCount);
    }
}

int main()
{
    int width = 224, height = 224;
    Pixel *image = (Pixel *)malloc(sizeof(Pixel) * width * height);
    GrayscalePixel *grayImage = (GrayscalePixel *)malloc(sizeof(GrayscalePixel) * width * height);

    read_cat_image(image, height, width);

    for (int i = 0; i < width * height; i++)
    {
        grayImage->V = 0;
    }

    Pixel *cudaImage;
    GrayscalePixel *cudaGrayscaleImage;
    cudaMalloc((void **)&cudaImage, sizeof(Pixel) * width * height);
    cudaMemcpy(cudaImage, image, sizeof(Pixel) * width * height, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&cudaGrayscaleImage, sizeof(GrayscalePixel) * width * height);

    dim3 dimGrid(ceil(width / 16.0), ceil(height / 16.0), 1);
    dim3 dimBlock(16, 16, 1);
    colorToGrayscaleKernel<<<dimGrid, dimBlock>>>(cudaGrayscaleImage, cudaImage, width, height);

    cudaMemcpy(grayImage, cudaGrayscaleImage, sizeof(GrayscalePixel) * width * height, cudaMemcpyDeviceToHost);

    // check the result
    int expected = image[287].R * 0.21f + image[287].G * 0.72f + image[287].B * 0.07f;
    int result = grayImage[287].V;
    printf("expected: %d, got: %d\n", expected, result);

    write_cat_image(grayImage, width, height, "gray.bin");

    GrayscalePixel *cudaBlurredImage;
    cudaMalloc((void **) &cudaBlurredImage, sizeof(GrayscalePixel) * width *height);
    blurGrayscaleKernel<<<dimGrid, dimBlock>>>(cudaGrayscaleImage, cudaBlurredImage, width, height);

    GrayscalePixel *blurredImage;
    blurredImage = (GrayscalePixel*) malloc(sizeof(GrayscalePixel) * width * height);
    cudaMemcpy(blurredImage, cudaBlurredImage, sizeof(GrayscalePixel) * width * height, cudaMemcpyDeviceToHost);

    write_cat_image(blurredImage, width, height, "blur.bin");

    cudaFree(cudaImage);
    cudaFree(cudaGrayscaleImage);
    cudaFree(cudaBlurredImage);

    free(image);
    free(grayImage);
    free(blurredImage);
}

void write_cat_image(GrayscalePixel *grayImage, int width, int height, const char filename[])
{
    FILE *ptrOut;
    ptrOut = fopen(filename, "wb");
    fwrite(grayImage, sizeof(GrayscalePixel), width * height, ptrOut);
    fclose(ptrOut);
}

void read_cat_image(Pixel *image, int height, int width)
{
    unsigned char buffer[224 * 224 * 3];
    FILE *ptr;
    ptr = fopen("cat.bin", "rb");
    fread(buffer, sizeof(buffer), 1, ptr);
    fclose(ptr);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int offset = y * width + x;
            image[offset].R = buffer[offset * 3];
            image[offset].G = buffer[offset * 3 + 1];
            image[offset].B = buffer[offset * 3 + 2];
        }
    }
}
