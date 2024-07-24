package ndgo

import (
	"fmt"
	"runtime"
	"sync"
)

// Operations that will be performed for some index i
// ----------------------------------------------------------------

var opAdd binOpFunc = func(a, b, res *Array, i int) {
	value := a.At(i) + b.At(i)
	res.Set(i, value)
}

var opMul binOpFunc = func(a, b, res *Array, i int) {
	value := a.At(i) * b.At(i)
	res.Set(i, value)
}

// Binary operations
// ------------------------------------------------------------------

// concurrent binary operation for given an operation function
func pBinOpArrays(a, b *Array, opfunc binOpFunc) *Array {
	res := NewArrayFromShape(a.Shape)

	n_routines := runtime.GOMAXPROCS(0)
	var chunk_size int = (res.Totalsize + n_routines - 1) / n_routines

	// add number of routines to a waitgroup
	var wg sync.WaitGroup
	wg.Add(n_routines)

	for r := 0; r < n_routines; r++ {
		start := r * chunk_size
		end := start + chunk_size
		if end > a.Totalsize {
			end = a.Totalsize
		}

		// start go routine in lambda function
		go func(s, e int) {
			defer wg.Done()

			// use linear indices as that will handle transpose and
			// non-contiguous arrays as well.
			for i := s; i < e; i++ {
				opfunc(a, b, res, i)
			}
		}(start, end)
	}

	wg.Wait()
	return res
}

func serialAddArrays(a, b *Array) *Array {
	res := NewArrayFromShape(a.Shape)

	// use linear indices as that will handle transpose and
	// non-contiguous arrays as well.
	for i := 0; i < res.Totalsize; i++ {
		opAdd(a, b, res, i)
	}

	return res
}

/*
add the elements of two Arrays elementwise
if the shapes are not equal but broadcastable,
then broadcasting will take place.
*/
func Add(a, b *Array) *Array {
	if CheckShapesEqual(a.Shape, b.Shape) {
		if a.Totalsize >= PARALLEL_BOUNDARY {
			return pBinOpArrays(a, b, opAdd)
		}
		return serialAddArrays(a, b)
	}

	res_shape, err := broadcastShapes(a.Shape, b.Shape)
	if err != nil {
		panic("AddError: cannot add arrays, shapes are not broadcastable")
	}

	afinal := broadcastArray(a, res_shape)
	bfinal := broadcastArray(b, res_shape)

	if afinal.Totalsize >= PARALLEL_BOUNDARY {
		return pBinOpArrays(afinal, bfinal, opAdd)
	}
	return serialAddArrays(afinal, bfinal)
}

// a - b
func Sub(a, b *Array) *Array {
	return Add(a, Neg(b))
}

// can be parallelized
func serialMulArrays(a, b *Array) *Array {
	res := NewArrayFromShape(a.Shape)

	// use linear indices as that will handle transpose and
	// non-contiguous arrays as well.
	for i := 0; i < res.Totalsize; i++ {
		opMul(a, b, res, i)
	}

	return res
}

/*
multiply the elements of two Arrays elementwise
if the shapes are not equal but broadcastable,
then broadcasting will take place.
*/
func Mul(a, b *Array) *Array {
	if CheckShapesEqual(a.Shape, b.Shape) {
		if a.Totalsize >= PARALLEL_BOUNDARY {
			return pBinOpArrays(a, b, opMul)
		}
		return serialMulArrays(a, b)
	}

	res_shape, err := broadcastShapes(a.Shape, b.Shape)
	if err != nil {
		panic("MulError: cannot multiply arrays, shapes are not broadcastable")
	}

	afinal := broadcastArray(a, res_shape)
	bfinal := broadcastArray(b, res_shape)

	if afinal.Totalsize >= PARALLEL_BOUNDARY {
		return pBinOpArrays(afinal, bfinal, opMul)
	}
	return serialMulArrays(afinal, bfinal)
}

/*
matrix multiplication of n-dimensional arrays.

for 2D arrays (m, n) @ (n, d) = (m, d)
for nD arrays (..., m, n) @ (..., n, d) = (..., m, d)

when the dimensions of arrays are greater than 2, we do N matmuls
on the last two axes of the operands. These N matmuls will be stacked
in the shape of the higher dimensions.
*/
func Matmul(a, b *Array) *Array {
	if a.Ndim < 2 || b.Ndim < 2 {
		panic(">> MatmulError: both arrays must have at least 2 dimensions for matmul.")
	}
	if a.Shape[a.Ndim-1] != b.Shape[b.Ndim-2] {
		panic(">> MatmulError: last dimension of first array must match second-last dimension of second array.")
	}

	// broadcast result shape untill last two axes
	a_shape_head := a.Shape[:a.Ndim-2]
	b_shape_head := b.Shape[:b.Ndim-2]

	res_shape_head, err := broadcastShapes(a_shape_head, b_shape_head)
	if err != nil {
		panic(fmt.Sprintf(">> MatmulError: %v", err))
	}

	result_shape := append(res_shape_head, a.Shape[a.Ndim-2], b.Shape[b.Ndim-1])

	result := NewArrayFromShape(result_shape)

	m := a.Shape[a.Ndim-2]
	n := a.Shape[a.Ndim-1]
	p := b.Shape[b.Ndim-1]

	totalops := result.Totalsize / (m * p)
	idxs := arrayIndicesFromShape(result_shape[:len(result_shape)-2])

	for idx := 0; idx < totalops; idx++ {
		nd_index := idxs.Indices[idx]
		// perform matmul for this slice
		// note: can be parallelized
		for i := 0; i < m; i++ {
			for j := 0; j < p; j++ {
				sum := float32(0.)
				for k := 0; k < n; k++ {
					// linear 1D index for a and b
					a_index1d, b_index1d := 0, 0
					// higher dimensions
					for d := 0; d < a.Ndim-2; d++ {
						a_index1d += (nd_index[d] * a.Strides[d])
					}
					for d := 0; d < b.Ndim-2; d++ {
						b_index1d += (nd_index[d] * b.Strides[d])
					}
					// last 2 dimensions
					a_index1d += (i*a.Strides[a.Ndim-2] + k*a.Strides[a.Ndim-1])
					b_index1d += (k*b.Strides[b.Ndim-2] + j*b.Strides[b.Ndim-1])

					sum += a.Data[a_index1d/a.Itemsize] * b.Data[b_index1d/b.Itemsize]
				}
				// same as a and b, for result
				r_index1d := 0
				for d := 0; d < a.Ndim-2; d++ {
					r_index1d += (nd_index[d] * result.Strides[d])
				}
				r_index1d += (i*result.Strides[result.Ndim-2] + j*result.Strides[result.Ndim-1])

				result.Data[r_index1d/result.Itemsize] = sum
			}
		}
	}
	return result
}
