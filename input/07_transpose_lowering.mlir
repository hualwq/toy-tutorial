func.func @transpose(%input: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %0 = "toy.transpose"(%input) {permutation = [1, 0]} : (tensor<2x3xf32>) -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}
