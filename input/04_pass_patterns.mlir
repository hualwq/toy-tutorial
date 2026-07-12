func.func @fold_transpose_and_reshape(%input: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "toy.transpose"(%input) {permutation = [1, 0]} : (tensor<2x3xf32>) -> tensor<3x2xf32>
  %1 = "toy.transpose"(%0) {permutation = [1, 0]} : (tensor<3x2xf32>) -> tensor<2x3xf32>
  %2 = "toy.reshape"(%1) {shape = [6]} : (tensor<2x3xf32>) -> tensor<6xf32>
  %3 = "toy.reshape"(%2) {shape = [2, 3]} : (tensor<6xf32>) -> tensor<2x3xf32>
  return %3 : tensor<2x3xf32>
}
