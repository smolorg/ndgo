package main

import (
	"fmt"
	ng "ndgo/ndgo"
)

func main() {
	a := ng.Arange(1, 4, 1).Reshape([]int{3, 1})
	b := ng.Arange(1, 3, 1).Reshape([]int{2})

	c := ng.Add(a, b)

	ng.PrettyPrint(a)
	fmt.Printf("\nTensor A's shape: ")
	fmt.Println(a.Shape)

	ng.PrettyPrint(b)
	fmt.Printf("\nTensor B's shape: ")
	fmt.Println(b.Shape)

	ng.PrettyPrint(c)
	fmt.Printf("Tensor C's shape: ")
	fmt.Println(c.Shape)
}
