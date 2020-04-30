# Outline of the PnP, based on algebraic minimization
"""
Input: A set of m 3D points: {xk} in homogeneous coordinates,m≥6
Input: A set of m corresponding image points{yk}, in C-normalized homogeneous coordinates
Output: (R| t), minimizing the algebraic error corresponding to Equation (15.35)

1   foreach k=1, . . . , m do 
2       foreach row∗ r_l ∈ [yk]_x do 
3           a = vectorization of the 3×4 matrix r_l x_k^T, as a row vector
4           Append this row at the bottom of A: A = [A;a]
5       end
6   end
7   Determine C0 from data matrix A:
8       Use either the homogeneous method, or the inhomogeneous method, to find c0 in the null space ofA
9       Reshape the vector c0 to 3×4 matrix C0 
10  Constraint enforcement of C0 = (A|b):
11      Set τ = sign(det(A))
12      SVD: τ*A = U @ S @ V^T 
13      Set R = U @ V^T, λ = 3 * τ / trace(S), t = λ * b
14      Return C = (R | t) 
15 ∗Possibly, use only two of the rows in [yk]_x since the 3 rows are linearly dependent
"""
