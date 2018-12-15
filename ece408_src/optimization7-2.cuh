#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define TILE_WIDTH 16
__constant__ float kernel[2744];

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

__global__ void forward_kernel(float *__restrict__ y, const float *__restrict__ x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */
// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]	

#define MIN( a, b ) ( (a < b) ? a : b )

//    printf("\n\n In kernel \n\n");

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    int W_grid = ceil(1.0*W_out/16.0);

    int b, m, w, h, w0, h0, h_base, w_base;

    __shared__ float x_tile[TILE_WIDTH][TILE_WIDTH];

    b=blockIdx.x; m=blockIdx.y; w0=threadIdx.x; h0=threadIdx.y;
    h_base=(blockIdx.z/W_grid)*TILE_WIDTH; // start
    w_base=(blockIdx.z % W_grid)*TILE_WIDTH; // start

    h=h_base+h0; w=w_base+w0; // ends

    float Pvalue=0;

    for (int c=0; c<C; c++){
	if (w<W_out && h<H_out){
		x_tile[h0][w0] = x4d(b, c, h+K/2, w+K/2);
	}
	else{
		x_tile[h0][w0]=0;
	}
	__syncthreads();
	
	for (int i=0;i<K;i++){
		for (int j=0;j<K;j++){
			if ((w+i)<W && (h+j)<H){
				if ((w + i) >= (w_base + K/2) && (w + i) < MIN((w_base  +TILE_WIDTH+ K/2), W_out + K/2) \
						&& (h + j) >= (h_base + K/2) && (h + j) < MIN((h_base + TILE_WIDTH + K/2), H_out + K/2)){
					Pvalue+=x_tile[h0 + j - K/2][w0 + i - K/2] * k4d(m, c, j, i);
				}
				else{
					Pvalue+=x4d(b,c,(h+j),(w+i)) * k4d(m,c,j,i);
				}
			}
		}
	}
	__syncthreads();
    }
    if (w<W_out && h<H_out){
	    y4d(b,m,h,w)=Pvalue;
    }

#undef y4d
#undef x4d
#undef k4d
#undef MIN
}

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    // CHECK_EQ(0, 1) << "Remove this line and replace with your implementation";

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
   // printf("\n\n\n\n %d %d %d %d %d %d \n\n\n\n", B, M, C, H, W, K);
    //  100 24 12 33 33 7
    //  100 12 1 72 72 7

    const int W_grid = ceil(1.*W_out/16.0);
    const int H_grid = ceil(1.*H_out/16.0);
    const int Z = W_grid * H_grid;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(B, M, Z);

    // Call the kernel
//    forward_kernel<<<gridDim, blockDim, sizeof(float)*((TILE_WIDTH+K-1)*(TILE_WIDTH+K-1)+K * K)>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
//    forward_kernel<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
//    forward_kernel<<<1, 1>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

    cudaMemcpyToSymbol(kernel, w.dptr_, sizeof(float) * K * K * K, 0, cudaMemcpyHostToDevice);
    forward_kernel<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
