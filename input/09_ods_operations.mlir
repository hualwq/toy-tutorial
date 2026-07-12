// Run with:
//   ../build/ods-toy-opt 09_ods_operations.mlir --verify-each
// The compact syntax is generated from `assemblyFormat` in OdsToyOps.td.
module {
  %x = arith.constant 2.0 : f32
  %y = arith.constant 3.0 : f32
  %scale = arith.constant 4.0 : f32
  %sum = ods_toy.add %x, %y : f32
  %product = ods_toy.multiply %sum, %scale : f32
  // `factor` 被省略了，所以相当于一个恒等变换
  %result = ods_toy.scale %product : f32
}
