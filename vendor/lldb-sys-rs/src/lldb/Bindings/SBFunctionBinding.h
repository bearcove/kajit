//===-- SBFunctionBinding.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBFunctionBinding_h_
#define LLDB_SBFunctionBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBFunctionRef CreateSBFunction(void);

LLDB_API SBFunctionRef CloneSBFunction(SBFunctionRef instance);

LLDB_API void DisposeSBFunction(SBFunctionRef instance);

LLDB_API bool SBFunctionIsValid(SBFunctionRef instance);

LLDB_API const char *SBFunctionGetName(SBFunctionRef instance);

LLDB_API const char *SBFunctionGetDisplayName(SBFunctionRef instance);

LLDB_API const char *SBFunctionGetMangledName(SBFunctionRef instance);

LLDB_API SBInstructionListRef SBFunctionGetInstructions(SBFunctionRef instance,
                                                        SBTargetRef target);

LLDB_API SBInstructionListRef SBFunctionGetInstructions2(SBFunctionRef instance,
                                                         SBTargetRef target,
                                                         const char *flavor);

LLDB_API SBAddressRef SBFunctionGetStartAddress(SBFunctionRef instance);

LLDB_API SBAddressRef SBFunctionGetEndAddress(SBFunctionRef instance);

LLDB_API uint32_t SBFunctionGetPrologueByteSize(SBFunctionRef instance);

LLDB_API SBTypeRef SBFunctionGetType(SBFunctionRef instance);

LLDB_API SBBlockRef SBFunctionGetBlock(SBFunctionRef instance);

LLDB_API ENUM(LanguageType) SBFunctionGetLanguage(SBFunctionRef instance);

LLDB_API bool SBFunctionGetIsOptimized(SBFunctionRef instance);

LLDB_API bool SBFunctionGetDescription(SBFunctionRef instance,
                                       SBStreamRef description);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBFunctionBinding_h_
