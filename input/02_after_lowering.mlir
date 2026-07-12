// Expected output of:
//   ../build/toy-opt input/01_toy_high_level.mlir -lower-toy-to-arith
//
// The SSA values (%x, %y, %sum, ...) and function structure are retained.
// Only the operation vocabulary changed from the toy dialect to arith.
module {
  func.func @scaled_sum(%x: f32, %y: f32, %scale: f32) -> f32 {
    %sum = arith.addf %x, %y : f32
    %result = arith.mulf %sum, %scale : f32
    return %result : f32
  }
}
