func.func private @twice(%x: f32) -> f32 {
  %0 = arith.addf %x, %x : f32
  return %0 : f32
}

func.func @caller(%arg0: f32) -> f32 {
  %0 = call @twice(%arg0) : (f32) -> f32
  return %0 : f32
}
