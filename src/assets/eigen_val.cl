#define EPSILON 1e-6

__kernel void jacobi(__global float* A, __global float* eigenvalues, int size) {
    int i = get_global_id(0);
    if (i >= size) return;

    int max_iter = 1000;
    for(int iter = 0; iter < max_iter; iter++) {
        // For each work item, handle a pair of rows/columns
        for(int j = 0; j < i; j++) {
            float Aii = A[i * size + i];
            float Ajj = A[j * size + j];
            float Aij = A[i * size + j];
            
            if (fabsf(Aij) < EPSILON) continue;
            
            float theta = 0.5f * atan2(2*Aij, Aii - Ajj);
            float cosTheta = cosf(theta);
            float sinTheta = sinf(theta);

            // Update elements
            A[i * size + i] = Aii*cosTheta*cosTheta + Ajj*sinTheta*sinTheta + 2*Aij*sinTheta*cosTheta;
            A[j * size + j] = Aii*sinTheta*sinTheta + Ajj*cosTheta*cosTheta - 2*Aij*sinTheta*cosTheta;
            A[i * size + j] = (Aii - Ajj)*sinTheta*cosTheta + Aij*(cosTheta*cosTheta - sinTheta*sinTheta);
            A[j * size + i] = A[i * size + j];
        }
    }

    // Extract eigenvalues
    for(int i = 0; i < size; i++) {
        eigenvalues[i] = A[i * size + i];
    }
}



