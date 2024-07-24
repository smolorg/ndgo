package main

import (
	"testing"
	"time"

	ng "ndgo/ndgo"
)

func TestApply(t *testing.T) {
	a := ng.Random([]int{2, 4, 2})
	_ = ng.Exp(a)
}

func TestReshape(t *testing.T) {
	_ = ng.Arange(1, 17, 1).Reshape([]int{2, 2, 4})
}

func TestIndices(t *testing.T) {
	a := ng.Arange(1, 17, 1).Reshape([]int{2, 4, 2})

	t.Log("Testing n-dimensional indices...")
	t.Log(a.Idxs.Indices)

	t.Log("Testing 1-dimensional indices...")
	t.Log(a.Lidxs.Indices)
	if a.Data[a.Totalsize-1] != a.Data[a.Lidxs.Indices[a.Totalsize-1]] {
		panic("bad linear indices")
	}
}

func TestTranspose(t *testing.T) {
	t.Log("Testing 1D transpose...")
	a := ng.Arange(1, 17, 1)
	b := a.Transpose(nil)
	t.Log(b.Shape, b.Strides, b.C_ORDER, b.F_ORDER)
	t.Log(b.Idxs, b.Lidxs)

	t.Log("Testing nD transpose...")
	a = ng.Arange(1, 17, 1).Reshape([]int{2, 4, 2})
	b = a.Transpose(nil)
	t.Log(b.Shape, b.Strides, b.C_ORDER, b.F_ORDER)
	t.Log(b.Idxs, b.Lidxs)
	ng.PrettyPrint(b)

	t.Log("Testing nD transpose with axes...")
	a = ng.Arange(1, 17, 1).Reshape([]int{2, 4, 2})
	b = a.Transpose([]int{2, 1, 0})
	t.Log(b.Shape, b.Strides, b.C_ORDER, b.F_ORDER)
	t.Log(b.Idxs, b.Lidxs)
	ng.PrettyPrint(b)
}

func TestNormalTransposeOps(t *testing.T) {
	a := ng.Arange(1, 17, 1).Reshape([]int{2, 4, 2})
	b := a.Transpose(nil)
	c := ng.Add(a, b)
	t.Log(c.Shape, c.Strides, c.C_ORDER, c.F_ORDER)
	t.Log(c.Idxs, c.Lidxs)
	ng.PrettyPrint(c)
}

func TestMatmul(t *testing.T) {
	a := ng.Arange(1, 17, 1).Reshape([]int{2, 4, 2})
	b := ng.Arange(1, 17, 1).Reshape([]int{2, 2, 4})
	c := ng.Matmul(a, b)
	t.Log(a.Shape, b.Shape, c.Shape)
	ng.PrettyPrint(c)
}

func TestParallelAdd(t *testing.T) {
	a := ng.Random([]int{100, 100})
	b := ng.Random([]int{100, 100})

	start := time.Now()
	_ = ng.Add(a, b)
	duration := time.Since(start)

	t.Log("Time taken for add: ", duration)
}

func TestSub(t *testing.T) {
	a := ng.Arange(1, 17, 1).Reshape([]int{2, 4, 2})
	b := ng.Arange(11, 27, 1).Reshape([]int{2, 4, 2})

	// subtract operation
	c := ng.Sub(a, b)

	ng.PrettyPrint(c)
}
