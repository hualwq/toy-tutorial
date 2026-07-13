// ============================================================
// linalg 基础：结构化线性代数操作
// ============================================================
// linalg 是 MLIR 中表达线性代数计算的核心方言
// 核心思想：用结构化 op（matmul / conv / generic）描述计算
// 替代手写 scf.for 循环
//
// 学习目标：
//   1. 理解 linalg.generic 的结构：indexing_maps + iterator_types
//   2. 理解 linalg.matmul 等命名 op
//   3. 理解 tensor 和 memref 两种内存模型
// ============================================================

// ----------------------------------------------------------
// 例 1: linalg.generic — 逐元素加法
// 最通用的 linalg op，用 region 描述计算逻辑
//
// 关键概念：
//   - indexing_maps: 每个 operand 的索引映射
//     (d0, d1) -> (d0, d1) 表示 i 维→i 维，j 维→j 维
//   - iterator_types: 各迭代维度的类型
//     "parallel" 表示该维可并行（无跨迭代依赖）
//   - ins / outs: 输入和输出张量
//     outs 提供初始值和形状，ins 是只读输入
// ----------------------------------------------------------
module {
  func.func @add(
    %A: tensor<4x4xf32>, %B: tensor<4x4xf32>
  ) -> tensor<4x4xf32> {
    %out = tensor.empty() : tensor<4x4xf32>

    // linalg.generic 是一个"结构化"的计算：
    // 编译器知道这是个二维并行操作，可以做 tiling/fusion/vectorize
    %C = linalg.generic {
      // 三个映射各对应：%A, %B, %out
      // (d0, d1) -> (d0, d1) 表示 operand 的 shape 和迭代空间一致
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,   // A[i][j]
        affine_map<(d0, d1) -> (d0, d1)>,   // B[i][j]
        affine_map<(d0, d1) -> (d0, d1)>    // C[i][j]
      ],
      iterator_types = ["parallel", "parallel"]
    }
    ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>)
    outs(%out : tensor<4x4xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %sum = arith.addf %a, %b : f32
      linalg.yield %sum : f32
    } -> tensor<4x4xf32>

    return %C : tensor<4x4xf32>
  }
}

// ----------------------------------------------------------
// 例 2: linalg.matmul — 矩阵乘法（命名 op）
// linalg.matmul 等价于下面的 generic：
//
//   linalg.generic {
//     indexing_maps = [
//       affine_map<(d0, d1, d2) -> (d0, d2)>,  // A[i][k]
//       affine_map<(d0, d1, d2) -> (d2, d1)>,  // B[k][j]
//       affine_map<(d0, d1, d2) -> (d0, d1)>   // C[i][j]
//     ],
//     iterator_types = ["parallel", "parallel", "reduction"]
//   }
//
// "reduction" 表示 k 维是归约维（sum over k）
// 命名 op 的优点：编译器直接知道这是 matmul，可以做更专门优化
// ----------------------------------------------------------
module {
  func.func @matmul(
    %A: tensor<32x64xf32>, %B: tensor<64x48xf32>
  ) -> tensor<32x48xf32> {
    %out = tensor.empty() : tensor<32x48xf32>

    // 一行 = TVM 的 te.compute
    %C = linalg.matmul
      ins(%A, %B : tensor<32x64xf32>, tensor<64x48xf32>)
      outs(%out : tensor<32x48xf32>)
      -> tensor<32x48xf32>

    return %C : tensor<32x48xf32>
  }
}

// ----------------------------------------------------------
// 例 3: linalg.conv_2d — 二维卷积（命名 op）
// NCHW 格式：%input[N][C][H][W], %filter[F][C][KH][KW]
// ----------------------------------------------------------
module {
  func.func @conv2d(
    %input: tensor<1x3x32x32xf32>,
    %filter: tensor<16x3x3x3xf32>
  ) -> tensor<1x16x30x30xf32> {
    %out = tensor.empty() : tensor<1x16x30x30xf32>

    %result = linalg.conv_2d_nchw_fchw
      ins(%input, %filter : tensor<1x3x32x32xf32>, tensor<16x3x3x3xf32>)
      outs(%out : tensor<1x16x30x30xf32>)
      -> tensor<1x16x30x30xf32>

    return %result : tensor<1x16x30x30xf32>
  }
}

// ----------------------------------------------------------
// 例 4: linalg 中两个内存模型的对比
//
// 模型 A：tensor 模型（值语义）
//   - 每个 op 产生新的 SSA value
//   - 无副作用，方便做 fusion/reordering 等变换
//   - 参考例 1-3 中的用法
//
// 模型 B：memref 模型（内存语义）
//   - 直接操作内存，有副作用
//   - 更接近底层，但限制了变换空间
//
// 推荐在 tensor 层做变换，最后 bufferize 到 memref
// ----------------------------------------------------------
module {
  // tensor 版本（值语义）
  func.func @matmul_tensor(
    %A: tensor<16x16xf32>, %B: tensor<16x16xf32>
  ) -> tensor<16x16xf32> {
    %out = tensor.empty() : tensor<16x16xf32>
    %C = linalg.matmul
      ins(%A, %B : tensor<16x16xf32>, tensor<16x16xf32>)
      outs(%out : tensor<16x16xf32>)
      -> tensor<16x16xf32>
    return %C : tensor<16x16xf32>
  }

  // memref 版本（内存语义）
  func.func @matmul_memref(
    %A: memref<16x16xf32>, %B: memref<16x16xf32>, %C: memref<16x16xf32>
  ) {
    linalg.matmul
      ins(%A, %B : memref<16x16xf32>, memref<16x16xf32>)
      outs(%C : memref<16x16xf32>)
    return
  }
}

// ============================================================
// 总结
// ============================================================
// linalg.generic：通用结构化 op，用 region 描述计算
// linalg.matmul / conv_2d：命名 op，直接用
//
// 共同点：
//   - 用 indexing_maps 描述张量访问模式
//   - 用 iterator_types 描述迭代语义（parallel/reduction）
//   - 保持结构化信息，方便上层做变换
//
// 运行验证（mlir-opt --verify-each）：
//   mlir-opt 01_linalg_basics.mlir --verify-each
// ============================================================
