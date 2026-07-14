// A minimal Operation Definition Specification (ODS) dialect.
//
// Run `cmake --build build --target OdsToyIncGen` to inspect the generated
// OdsToyOps.h.inc and OdsToyOps.cpp.inc files in the build directory.

#ifndef ODS_TOY_OPS
#define ODS_TOY_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "ods/OdsToyCostOpInterface.td"

// 定义一个最小的 Dialect，名字为 ods_toy，命名空间为 ::ods_toy
def ODS_ToyDialect : Dialect {
  let name = "ods_toy";
  let cppNamespace = "::ods_toy";
  let summary = "Minimal dialect used to study MLIR ODS";
}

// 创建一个 Operation 模板类，Op名称为 mnemonic， 模板参数列表为空
class ODS_ToyOp<string mnemonic, list<Trait> traits = []> :
    Op<ODS_ToyDialect, mnemonic, traits>;

// 创建一个类实例，Op名称为 add，SameOperandsAndResultType 表示operand 和 result 类型相同，Pure 表示没有副作用
def ODS_Toy_AddOp : ODS_ToyOp<"add", [Pure, SameOperandsAndResultType,
    DeclareOpInterfaceMethods<ODS_ToyCostOpInterface>]> {
  let summary = "Add two f32 scalar values";
  // ins 表示输入参数，outs 表示输出参数
  let arguments = (ins F32:$lhs, F32:$rhs);
  let results = (outs F32:$result);
  // 定义 MLIR 文本格式
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def ODS_Toy_MultiplyOp : ODS_ToyOp<"multiply", [Pure,
    SameOperandsAndResultType,
    DeclareOpInterfaceMethods<ODS_ToyCostOpInterface>]> {
  let summary = "Multiply two f32 scalar values";
  let arguments = (ins F32:$lhs, F32:$rhs);
  let results = (outs F32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def ODS_Toy_ScaleOp : ODS_ToyOp<"scale", [Pure,
                                            SameOperandsAndResultType]> {
  let summary = "Scale an f32 value by a factor";
  // 设置 factor 的默认值为 1.0f
  // `ins` dag 同时描述 operand 和 attribute；Attr 是编译期确定的常量。
  let arguments = (ins F32:$input,
    DefaultValuedAttr<F32Attr, "1.0f">:$factor);
  // `outs` dag 描述 operation 的 result。
  let results = (outs F32:$result);
  // OpBuilder 的 `ins` dag 描述 C++ builder 参数，而不是 IR type constraint。
  let builders = [
    OpBuilder<(ins "::mlir::Value":$input, "float":$factor), [{
      $_state.addOperands(input);
      $_state.addTypes(input.getType());
      $_state.addAttribute("factor", $_builder.getF32FloatAttr(factor));
    }]>
  ];
  // attr-dict prints `factor = ...` only when it is explicitly present.  When
  // absent, factor() materializes the DefaultValuedAttr default (1.0f).
  let assemblyFormat = "$input attr-dict `:` type($result)";
}

#endif // ODS_TOY_OPS

// 为什么只有 scale 写了 builder？
// 其余两个Op都有默认builder，scale有Attribute参数，默认值为1.0f，所以需要自定义builder来处理这个参数。
