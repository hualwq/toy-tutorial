// A standalone .td file prevents interface generation from re-emitting the
// unrelated standard interfaces included by OdsToyOps.td.

#ifndef ODS_TOY_COST_OP_INTERFACE
#define ODS_TOY_COST_OP_INTERFACE

include "mlir/IR/OpBase.td"

// An OpInterface is a capability contract. A generic pass can ask an
// Operation whether it implements this interface without knowing its Op class.
def ODS_ToyCostOpInterface : OpInterface<"ToyCostOpInterface"> {
  let cppNamespace = "::ods_toy";
  let description = "Queries an operation's estimated scalar cost";
  let methods = [
    InterfaceMethod<"Returns the operation's estimated cost", "int64_t",
                    "getCost", (ins)>
  ];
}

#endif // ODS_TOY_COST_OP_INTERFACE
