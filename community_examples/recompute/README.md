
This demo belongs to the Forward Recomputation Backpropagation Document.

Expected output of this demo is like:

```shell
name: "x"
type {
  type: LOD_TENSOR
  lod_tensor {
    tensor {
      data_type: FP32
      dims: -1
      dims: 32
    }
    lod_level: 0
  }
}
persistable: false

Finished FF
Finished optimize
step=0 cost=0.740719
step=1 cost=1.113626
step=2 cost=0.699762
step=3 cost=0.663064
step=4 cost=0.785782
step=5 cost=0.655717
step=6 cost=0.722237
step=7 cost=0.699711
step=8 cost=0.698443
step=9 cost=0.698395
```
