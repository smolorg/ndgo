package main

import (
	"fmt"
	ng "ndgo/ndgo"
)

func main() {
	a := ng.Arange(1, 17, 1).Reshape([]int{2, 4, 2})
	b := a.Transpose(nil)

	c := ng.Add(a, b)
	ng.PrettyPrint(c)
	fmt.Println(c.Shape)
}
