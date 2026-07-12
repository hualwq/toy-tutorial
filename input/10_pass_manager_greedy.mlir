// Each nested module is independently visited by the nested pipeline.
module {
  %x = arith.constant 42.0 : f32
  %zero = arith.constant 0.0 : f32
  %0 = arith.addf %x, %zero : f32
  %1 = arith.addf %0, %zero : f32
  %2 = arith.addf %1, %zero : f32
}

// The rule also handles zero on the left-hand side.
module {
  %x = arith.constant 7.0 : f32
  %zero = arith.constant 0.0 : f32
  %0 = arith.addf %zero, %x : f32
}
