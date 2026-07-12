func.func @infer_shapes(%input: tensor<2x3xf32>) {
  %0 = "toy.transpose"(%input) {permutation = [1, 0]} : (tensor<2x3xf32>) -> tensor<?x?xf32>
  %1 = "toy.reshape"(%0) {shape = [6]} : (tensor<?x?xf32>) -> tensor<?xf32>
  "toy.reshape"(%1) {shape = [2, 3]} : (tensor<?xf32>) -> tensor<?x?xf32>
  return
}
