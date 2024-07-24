package ndgo

import (
	"math"
)

// -1*x for all x in an Array
func neg() ArrayFunc {
	return func(x float32) float32 {
		ans := -1 * x
		return ans
	}
}

// e**x for all x in an Array
func exp() ArrayFunc {
	return func(x float32) float32 {
		ans := math.Exp(float64(x))
		return float32(ans)
	}
}

// scale the Array using a scalar
func log() ArrayFunc {
	return func(x float32) float32 {
		ans := math.Log(float64(x))
		return float32(ans)
	}
}

// sin(x) for all x in an Array
func sin() ArrayFunc {
	return func(x float32) float32 {
		ans := math.Sin(float64(x))
		return float32(ans)
	}
}

// cos(x) for all x in an Array
func cos() ArrayFunc {
	return func(x float32) float32 {
		ans := math.Cos(float64(x))
		return float32(ans)
	}
}

// tan(x) for all x in an Array
func tan() ArrayFunc {
	return func(x float32) float32 {
		ans := math.Tan(float64(x))
		return float32(ans)
	}
}

// tanh(x) for all x in an Array
func tanh() ArrayFunc {
	return func(x float32) float32 {
		ans := math.Tanh(float64(x))
		return float32(ans)
	}
}

// sigmoid(x) for all x in an Array
func sigmoid() ArrayFunc {
	return func(x float32) float32 {
		xn := float64(x)
		ans := (1 / (1 + math.Exp(-1*xn)))
		return float32(ans)
	}
}
