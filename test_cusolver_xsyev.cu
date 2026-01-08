#include <stdio.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

int main() {
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);

    int batchSize = 2;
    int n = 4;
    cudaDataType dataType = CUDA_R_32F;
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

    size_t workspaceInBytes = 0;

    cusolverStatus_t status =
        cusolverDnXsyevBatched_BufferSize(
            handle,
            jobz,
            uplo,
            n,
            dataType,
            nullptr,
            n,
            dataType,
            nullptr,
            nullptr,
            dataType,
            &workspaceInBytes,
            batchSize
        );

    if (status != CUSOLVER_STATUS_SUCCESS) {
        printf("cusolverDnXsyevBatched_BufferSize failed: %d\n", status);
    } else {
        printf("SUCCESS: workspace size = %zu bytes\n", workspaceInBytes);
    }

    cusolverDnDestroy(handle);
    return 0;
}
#include <stdio.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

int main() {
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);

    int batchSize = 2;
    int n = 4;
    cudaDataType dataType = CUDA_R_32F;
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

    size_t workspaceInBytes = 0;

    cusolverStatus_t status =
        cusolverDnXsyevBatched_BufferSize(
            handle,
            jobz,
            uplo,
            n,
            dataType,
            nullptr,
            n,
            dataType,
            nullptr,
            nullptr,
            dataType,
            &workspaceInBytes,
            batchSize
        );

    if (status != CUSOLVER_STATUS_SUCCESS) {
        printf("cusolverDnXsyevBatched_BufferSize failed: %d\n", status);
    } else {
        printf("SUCCESS: workspace size = %zu bytes\n", workspaceInBytes);
    }

    cusolverDnDestroy(handle);
    return 0;
}
