//===-- SBSymbolBinding.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBSymbolBinding_h_
#define LLDB_SBSymbolBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBSymbolRef CreateSBSymbol(void);

LLDB_API SBSymbolRef CloneSBSymbol(SBSymbolRef instance);

LLDB_API void DisposeSBSymbol(SBSymbolRef instance);

LLDB_API bool SBSymbolIsValid(SBSymbolRef instance);

LLDB_API const char *SBSymbolGetName(SBSymbolRef instance);

LLDB_API const char *SBSymbolGetDisplayName(SBSymbolRef instance);

LLDB_API const char *SBSymbolGetMangledName(SBSymbolRef instance);

LLDB_API SBInstructionListRef SBSymbolGetInstructions(SBSymbolRef instance,
                                                      SBTargetRef target);

LLDB_API SBInstructionListRef SBSymbolGetInstructions2(
    SBSymbolRef instance, SBTargetRef target, const char *flavor_string);

LLDB_API SBAddressRef SBSymbolGetStartAddress(SBSymbolRef instance);

LLDB_API SBAddressRef SBSymbolGetEndAddress(SBSymbolRef instance);

LLDB_API uint32_t SBSymbolGetPrologueByteSize(SBSymbolRef instance);

LLDB_API ENUM(SymbolType) SBSymbolGetType(SBSymbolRef instance);

LLDB_API bool SBSymbolGetDescription(SBSymbolRef instance,
                                     SBStreamRef description);

LLDB_API bool SBSymbolIsExternal(SBSymbolRef instance);

LLDB_API bool SBSymbolIsSynthetic(SBSymbolRef instance);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBSymbolBinding_h_
