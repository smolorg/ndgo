package ndgo

import (
	"fmt"
	"math"
	"math/rand"
)

const SIZEOF_FLOAT32 int = 4
const PARALLEL_BOUNDARY int = 1e5

// holds all nD indices of the array
type ArrayIndices struct {
	Indices [][]int
	Count   int
}

// holds the 1D indices corresponding to nD indices
type LinearIndices struct {
	Indices []int
	Count   int
}

type Array struct {
	Data        []float32
	Shape       []int
	Strides     []int
	Backstrides []int
	Ndim        int
	Itemsize    int
	Totalsize   int
	Idxs        *ArrayIndices
	Lidxs       *LinearIndices
	C_ORDER     bool
	F_ORDER     bool
}

type ArrayFunc func(float32) float32
type binOpFunc func(*Array, *Array, *Array, int)

// private functions
// ------------------------------------------------------------

// strides for an Array
func (arr *Array) recalculateStrides() {
	arr.Strides[arr.Ndim-1] = arr.Itemsize
	for i := arr.Ndim - 2; i >= 0; i-- {
		arr.Strides[i] = arr.Strides[i+1] * arr.Shape[i+1]
	}
}

// backstrides for an Array
func (arr *Array) recalculateBackstrides() {
	for i := arr.Ndim - 1; i >= 0; i-- {
		arr.Backstrides[i] = -1 * arr.Strides[i] * (arr.Shape[i] - 1)
	}
}

// nD indices for an Array
func (arr *Array) createArrayIndices() {
	arr.Idxs = arrayIndicesFromShape(arr.Shape)
}

func arrayIndicesFromShape(shape []int) *ArrayIndices {
	ndim := len(shape)
	totalsize := 1
	for _, v := range shape {
		totalsize *= v
	}

	idxs := &ArrayIndices{
		Count:   totalsize,
		Indices: make([][]int, totalsize),
	}

	for i := range idxs.Indices {
		idxs.Indices[i] = make([]int, ndim)
	}

	currentIndex := make([]int, ndim)
	for i := 0; i < idxs.Count; i++ {
		copy(idxs.Indices[i], currentIndex)

		// increment the index
		for j := ndim - 1; j >= 0; j-- {
			if currentIndex[j]+1 < shape[j] {
				currentIndex[j]++
				break
			}
			currentIndex[j] = 0
		}
	}

	return idxs
}

// 1D equivalent of nD indices
func (arr *Array) createLinearIndices() {
	arr.Lidxs = &LinearIndices{
		Count:   arr.Totalsize,
		Indices: make([]int, arr.Totalsize),
	}

	for i := 0; i < arr.Totalsize; i++ {
		arr.Lidxs.Indices[i] = 0
		for j := 0; j < arr.Ndim; j++ {
			arr.Lidxs.Indices[i] += (arr.Idxs.Indices[i][j] * arr.Strides[j]) / SIZEOF_FLOAT32
		}
	}
}

// setArrayFlags sets flags for array
func (arr *Array) setArrayFlags() {
	arr.C_ORDER = arr.Strides[arr.Ndim-1] == arr.Itemsize
	arr.F_ORDER = arr.Strides[0] == arr.Itemsize
}

func getRandom(min, max float32) float32 {
	return min + rand.Float32()*(max-min)
}

func getRandomInt(min, max int) int {
	return min + rand.Intn(max-min+1)
}

func checkShapeCompatible(arr *Array, shape []int) bool {
	var size_new int = 1
	for _, value := range shape {
		size_new *= value
	}

	return (arr.Totalsize == size_new)
}

// Public functions
// ------------------------------------------------------

func NewArrayFromShape(shape []int) *Array {
	ndim := len(shape)
	if ndim <= 0 {
		panic(fmt.Sprintf("Cannot initialize Array of dimensions %d", ndim))
	}

	arr := &Array{
		Ndim:        ndim,
		Shape:       make([]int, ndim),
		Strides:     make([]int, ndim),
		Backstrides: make([]int, ndim),
		Itemsize:    4, // size of float32
	}

	arr.Totalsize = 1
	for i := 0; i < arr.Ndim; i++ {
		arr.Shape[i] = shape[i]
		arr.Totalsize *= shape[i]
	}

	arr.Data = make([]float32, arr.Totalsize)
	arr.recalculateStrides()
	arr.recalculateBackstrides()
	arr.createArrayIndices()
	arr.createLinearIndices()
	arr.setArrayFlags()

	return arr
}

// returns the element at the linear index specified by i
func (arr *Array) At(i int) float32 {
	return arr.Data[arr.Lidxs.Indices[i]]
}

// sets the element at linear index i, by the given value
func (arr *Array) Set(i int, value float32) {
	arr.Data[arr.Lidxs.Indices[i]] = value
}

// Random creates a random array of floats from shape,
// values will be in range [0.0, 1.0]
func Random(shape []int) *Array {
	arr := NewArrayFromShape(shape)
	for i := range arr.Data {
		arr.Set(i, getRandom(0, 1))
	}
	return arr
}

// RandomInts creates a random int array from shape, values will be in range [min, max)
func RandomInts(shape []int, min, max int) *Array {
	if min >= max {
		panic(fmt.Sprintf("Value of min %d must be less than value of max %d", min, max))
	}
	arr := NewArrayFromShape(shape)
	for i := range arr.Data {
		arr.Set(i, float32(getRandomInt(min, max)))
	}
	return arr
}

// Arange creates an array with values from start to end (exclusive) with the given step
func Arange(start, end, step float32) *Array {
	if start >= end {
		panic("Start value should be less than end value")
	}
	if step <= 0 {
		panic("Step value should be greater than 0")
	}

	length := int(math.Ceil(float64((end - start) / step)))
	shape := []int{length}
	arr := NewArrayFromShape(shape)

	curr := start
	for i := range arr.Data {
		arr.Set(i, curr)
		curr += step
	}

	return arr
}

func traverseHelper(data []float32, shape, strides, backstrides []int, ndim, depth, offset int) int {
	// we are at the last dimension
	if depth == ndim-1 {
		fmt.Printf("[")
		for i := 0; i < shape[ndim-1]; i++ {
			if i != shape[ndim-1]-1 {
				fmt.Printf("%.3f ", data[offset])
			} else {
				fmt.Printf("%.3f", data[offset])
			}
			if i != shape[ndim-1]-1 {
				offset += (strides[ndim-1] / SIZEOF_FLOAT32)
			}
		}
		fmt.Printf("]")
		// backstep
		offset += (backstrides[ndim-1] / SIZEOF_FLOAT32)
		return offset
	}

	if depth == 0 {
		fmt.Printf("[\n")
	} else {
		fmt.Printf("[")
	}
	offset = traverseHelper(data, shape, strides, backstrides, ndim, depth+1, offset)
	for i := 0; i < shape[depth]-1; i++ {
		offset += (strides[depth] / SIZEOF_FLOAT32)
		fmt.Println()
		offset = traverseHelper(data, shape, strides, backstrides, ndim, depth+1, offset)
	}
	if depth == 0 {
		fmt.Printf("]\n")
	} else {
		fmt.Printf("]")
	}
	offset += (backstrides[depth] / SIZEOF_FLOAT32)
	if depth != 0 {
		fmt.Println()
	}
	return offset
}

// prints the array similar to numpy
func PrettyPrint(arr *Array) {
	traverseHelper(arr.Data, arr.Shape, arr.Strides, arr.Backstrides, arr.Ndim, 0, 0)
}

// can be parallelized
func pApply(arr *Array, fun ArrayFunc) {
	for i := 0; i < arr.Totalsize; i++ {
		arr.Set(i, fun(arr.Data[i]))
	}
}

// applies an ArrayFunc to all the elements of an Array
// and returns a new Array
func Apply(arr *Array, fun ArrayFunc) *Array {
	if fun == nil {
		panic("ApplyError: function argument nil/missing.")
	}

	res := NewArrayFromShape(arr.Shape)
	res.FromValues(arr.Data)
	pApply(res, fun)
	return res
}

// applies an ArrayFunc to all the elements of an Array
// in-place, and DOES NOT return a new Array
func Apply_(arr *Array, fun ArrayFunc) {
	if fun == nil {
		panic("ApplyError: function argument nil/missing.")
	}

	pApply(arr, fun)
}
