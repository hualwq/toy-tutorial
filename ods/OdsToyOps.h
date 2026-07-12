#ifndef TOY_TUTORIAL_ODS_TOY_OPS_H
#define TOY_TUTORIAL_ODS_TOY_OPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "OdsToyCostOpInterface.h"

#define GET_OP_CLASSES
#include "OdsToyOps.h.inc"

#endif // TOY_TUTORIAL_ODS_TOY_OPS_H
