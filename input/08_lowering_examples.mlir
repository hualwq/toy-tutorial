// All three buffers use a static shape so the Affine lowering has a static
// loop bound. The operation computes: out[i] = a * x[i] + y[i].
func @saxpy(%a: f32, %x: memref<8xf32>, %y: memref<8xf32>,
            %out: memref<8xf32>) {
  "toy.saxpy"(%a, %x, %y, %out) :
    (f32, memref<8xf32>, memref<8xf32>, memref<8xf32>) -> ()
  return
}


// module attributes {llvm.data_layout = ""} {
//   llvm.func @saxpy(%arg0: f32, %arg1: !llvm.ptr<f32>, %arg2: !llvm.ptr<f32>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: !llvm.ptr<f32>, %arg7: !llvm.ptr<f32>, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: !llvm.ptr<f32>, %arg12: !llvm.ptr<f32>, %arg13: i64, %arg14: i64, %arg15: i64) {
//     %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
//     %1 = llvm.insertvalue %arg1, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
//     %2 = llvm.insertvalue %arg2, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
//     %3 = llvm.insertvalue %arg3, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
//     %4 = llvm.insertvalue %arg4, %3[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
//     %5 = llvm.insertvalue %arg5, %4[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
//     %6 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
//     %7 = llvm.insertvalue %arg6, %6[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
//     %8 = llvm.insertvalue %arg7, %7[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
//     %9 = llvm.insertvalue %arg8, %8[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
//     %10 = llvm.insertvalue %arg9, %9[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
//     %11 = llvm.insertvalue %arg10, %10[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
//     %12 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
//     %13 = llvm.insertvalue %arg11, %12[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
//     %14 = llvm.insertvalue %arg12, %13[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
//     %15 = llvm.insertvalue %arg13, %14[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
//     %16 = llvm.insertvalue %arg14, %15[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
//     %17 = llvm.insertvalue %arg15, %16[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
//     %18 = llvm.mlir.constant(0 : index) : i64
//     %19 = llvm.mlir.constant(8 : index) : i64
//     %20 = llvm.mlir.constant(1 : index) : i64
//     llvm.br ^bb1(%18 : i64)
//   ^bb1(%21: i64):  // 2 preds: ^bb0, ^bb2
//     %22 = llvm.icmp "slt" %21, %19 : i64
//     llvm.cond_br %22, ^bb2, ^bb3
//   ^bb2:  // pred: ^bb1
//     %23 = llvm.getelementptr %arg2[%21] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
//     %24 = llvm.load %23 : !llvm.ptr<f32>
//     %25 = llvm.getelementptr %arg7[%21] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
//     %26 = llvm.load %25 : !llvm.ptr<f32>
//     %27 = llvm.fmul %arg0, %24  : f32
//     %28 = llvm.fadd %27, %26  : f32
//     %29 = llvm.getelementptr %arg12[%21] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
//     llvm.store %28, %29 : !llvm.ptr<f32>
//     %30 = llvm.add %21, %20  : i64
//     llvm.br ^bb1(%30 : i64)
//   ^bb3:  // pred: ^bb1
//     llvm.return
//   }
// }