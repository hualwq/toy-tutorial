// This is valid input for toy-opt. A single MLIR module may use several
// dialects at once; it is not necessary to have one dialect per IR level.
module {
  func.func @mixed(%x: f32, %y: f32) -> f32 {
    %sum = "toy.add"(%x, %y) : (f32, f32) -> f32
    %two = arith.constant 2.0 : f32
    %result = "toy.multiply"(%sum, %two) : (f32, f32) -> f32
    return %result : f32
  }
}
