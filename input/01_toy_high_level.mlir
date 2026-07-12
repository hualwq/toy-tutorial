// Run with:
//   ../build/toy-opt input/01_toy_high_level.mlir --verify-each
//
// `toy` owns the domain names. At this point the compiler still knows that
// these arithmetic operations came from the Toy source language.
module {
  func.func @scaled_sum(%x: f32, %y: f32, %scale: f32) -> f32 {
    %sum = "toy.add"(%x, %y) : (f32, f32) -> f32
    %result = "toy.multiply"(%sum, %scale) : (f32, f32) -> f32
    return %result : f32
  }
}
