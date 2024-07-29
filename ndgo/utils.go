package ndgo

import "errors"

/*
Check shapes equal
*/
func CheckShapesEqual(left, right []int) bool {
	ndim1 := len(left)
	ndim2 := len(right)

	var equal bool = true

	if ndim1 != ndim2 {
		equal = false
	} else {
		for i, lval := range left {
			if lval != right[i] {
				equal = false
				break
			}
		}
	}

	return equal
}

/*
Broadcast the two shapes to give a final shape
*/
func broadcastShapes(left, right []int) ([]int, error) {
	if CheckShapesEqual(left, right) {
		res_shape := make([]int, len(left))
		copy(res_shape, left)
		return res_shape, nil
	}

	lndim, rndim := len(left), len(right)
	var res_ndim int
	if lndim > rndim {
		res_ndim = lndim
	} else {
		res_ndim = rndim
	}

	// 1s to prepend
	l1add := res_ndim - lndim
	r1add := res_ndim - rndim

	lf_shape := make([]int, res_ndim)
	rf_shape := make([]int, res_ndim)
	res_shape := make([]int, res_ndim)

	// for left
	inda := 0
	for i := 0; i < res_ndim; i++ {
		if i < l1add {
			lf_shape[i] = 1
		} else {
			lf_shape[i] = left[inda]
			inda++
		}
	}

	// for right
	indb := 0
	for i := 0; i < res_ndim; i++ {
		if i < r1add {
			rf_shape[i] = 1
		} else {
			rf_shape[i] = right[indb]
			indb++
		}
	}

	for i := 0; i < res_ndim; i++ {
		if lf_shape[i] == 1 || rf_shape[i] == 1 || lf_shape[i] == rf_shape[i] {
			if lf_shape[i] > rf_shape[i] {
				res_shape[i] = lf_shape[i]
			} else {
				res_shape[i] = rf_shape[i]
			}
		} else {
			return nil, errors.New("shapes are not broadcastable")
		}
	}

	return res_shape, nil
}

/*
broadcasts an array given a new shape and its dimensions.

note: errors are not checked, and it is to be assumed that the Array
is "broadcastable" to the new shape.

if you use this function, you will have to manually free the result of broadcasted array
*/
func broadcastArray(arr *Array, shape []int) *Array {
	res := NewArrayFromShape(shape)

	n_prepend := len(shape) - arr.Ndim

	for i := 0; i < res.Totalsize; i++ {
		srcIdx := 0
		for dim := 0; dim < arr.Ndim; dim++ {
			// for dimensions which are not singleton,
			// use result's n-dimensional index to
			// calculate the 1D index in source array
			if arr.Shape[dim] > 1 {
				srcIdx += (res.Idxs.Indices[i][n_prepend+dim] % arr.Shape[dim]) * arr.Strides[dim]
			}
		}

		res.Data[res.Lidxs.Indices[i]] = arr.Data[srcIdx/arr.Itemsize]
	}

	return res
}

func isValidPermutation(slice []int, n int) bool {
	seen := make([]bool, n)

	for _, num := range slice {
		if num < 0 || num >= n {
			return false
		}
		// when the number repeats
		if seen[num] {
			return false
		}
		seen[num] = true
	}

	return true
}
