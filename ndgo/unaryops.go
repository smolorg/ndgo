package ndgo

// Unary operations
// --------------------------------------------------------------

// FromValues initializes the Array's data with values
func (arr *Array) FromValues(values []float32) {
	if len(values) != arr.Totalsize {
		panic("Values length must match Array's total size")
	}
	copy(arr.Data, values)
}

// Reshape an array to a new array according to new shape
func (arr *Array) Reshape(shape []int) *Array {
	var possible bool = checkShapeCompatible(arr, shape)
	if !possible {
		panic("ReshapeError: cannot reshape due to invalid given shape.")
	}

	var res *Array = NewArrayFromShape(shape)
	res.FromValues(arr.Data)

	return res
}

/*
transpose an Array along given permutation of axes,
here axes is a valid permutation of length equal to shape.
If axes is nil, then axes will be reversed.
*/
func (arr *Array) Transpose(axes []int) *Array {
	// check if axes is valid
	if axes != nil && !isValidPermutation(axes, arr.Ndim) {
		panic("TransposeError: axes must be nil or a valid permutation.")
	}

	res := NewArrayFromShape(arr.Shape)
	res.FromValues(arr.Data)
	if arr.Ndim == 1 {
		return res
	}

	_axes := make([]int, res.Ndim)
	if axes == nil {
		for i := 0; i < res.Ndim; i++ {
			_axes[i] = res.Ndim - 1 - i
		}
	}

	newshape := make([]int, res.Ndim)
	newstrides := make([]int, res.Ndim)
	for i := 0; i < res.Ndim; i++ {
		if axes != nil {
			newshape[i] = arr.Shape[axes[i]]
			newstrides[i] = arr.Strides[axes[i]]
		} else {
			newshape[i] = arr.Shape[_axes[i]]
			newstrides[i] = arr.Strides[_axes[i]]
		}
	}
	copy(res.Shape, newshape)
	copy(res.Strides, newstrides)
	res.recalculateBackstrides()
	res.createArrayIndices()
	res.createLinearIndices()
	res.setArrayFlags()

	return res
}

// Reshape an array inplace according to given shape
func (arr *Array) Reshape_(shape []int) {
	var possible bool = checkShapeCompatible(arr, shape)
	ndim := len(shape)

	if !possible || arr.Ndim < ndim || arr.Ndim > ndim {
		panic("ReshapeError: cannot reshape due to invalid given shape.")
	}

	arr.Ndim = ndim
	copy(arr.Shape, shape)

	arr.recalculateStrides()
	arr.recalculateBackstrides()
	arr.createArrayIndices()
	arr.setArrayFlags()
}

// Apply operations
// ---------------------------------------------------------------

func Neg(arr *Array) *Array {
	res := Apply(arr, neg())
	return res
}

// e**x for all x in the Array
func Exp(arr *Array) *Array {
	res := Apply(arr, exp())
	return res
}

// ln(x) for all x in the Array
func Log(arr *Array) *Array {
	res := Apply(arr, log())
	return res
}

func Sin(arr *Array) *Array {
	res := Apply(arr, sin())
	return res
}

func Cos(arr *Array) *Array {
	res := Apply(arr, cos())
	return res
}

func Tan(arr *Array) *Array {
	res := Apply(arr, tan())
	return res
}

func Tanh(arr *Array) *Array {
	res := Apply(arr, tanh())
	return res
}

func Sigmoid(arr *Array) *Array {
	res := Apply(arr, sigmoid())
	return res
}
