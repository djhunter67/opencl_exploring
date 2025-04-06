// matrix_ops.cl - OpenCL kernel for matrix operations related to eigenvalue calculation

__kernel void multiply_matrix_vector(
    __global float* A,
    __global float* x,
    __global float* y,
    int size)
{
    int i = get_global_id(0);
    
    if (i < size) {
        float sum = 0.0f;
        for (int j = 0; j < size; j++) {
            sum += A[i * size + j] * x[j];
        }
        y[i] = sum;
    }
}

__kernel void calculate_residual(
    __global float* Ax,
    __global float* lambda_x,
    __global float* residual,
    int size)
{
    int i = get_global_id(0);
    
    if (i < size) {
        residual[i] = Ax[i] - lambda_x[i];
    }
}
