# ndgo

N-dimensional array implementation in Go without any third party library, similar to numpy.

### Working

Currently, `ndgo` only supports N-d arrays of the `float32` datatype. That is, every element in the N-d array will be a `float32`. I did this because:

1. Integers can also be represented by floats (even though float takes more bytes in memory).
2. And for educational/learning purposes I just wanted to use float.
3. Support for more datatypes might be added in the future, might be.


Example usage (look at [play.go](play.go) file):

```go
package main

import (
    ng "ndgo/ndgo"
)

func main() {
    a := ng.Arange(1, 17, 1).Reshape([]int{2, 4, 2})
    b := a.Transpose(nil)

    c := ng.Add(a, b)
    ng.PrettyPrint(c)
    fmt.Println(c.Shape)
}
```

Output:

```
[
[[2.000 11.000]
[6.000 15.000]
[10.000 19.000]
[14.000 23.000]]

[[11.000 20.000]
[15.000 24.000]
[19.000 28.000]
[23.000 32.000]]
]
[2 4 2]
```