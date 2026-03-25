//===-- SBSymbolContextBinding.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBSymbolContextBinding_h_
#define LLDB_SBSymbolContextBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBSymbolContextRef CreateSBSymbolContext(void);

LLDB_API SBSymbolContextRef CloneSBSymbolContext(SBSymbolContextRef instance);

LLDB_API void DisposeSBSymbolContext(SBSymbolContextRef instance);

LLDB_API bool SBSymbolContextIsValid(SBSymbolContextRef instance);

LLDB_API SBModuleRef SBSymbolContextGetModule(SBSymbolContextRef instance);

LLDB_API SBCompileUnitRef
SBSymbolContextGetCompileUnit(SBSymbolContextRef instance);

LLDB_API SBFunctionRef SBSymbolContextGetFunction(SBSymbolContextRef instance);

LLDB_API SBBlockRef SBSymbolContextGetBlock(SBSymbolContextRef instance);

LLDB_API SBLineEntryRef
SBSymbolContextGetLineEntry(SBSymbolContextRef instance);

LLDB_API SBSymbolRef SBSymbolContextGetSymbol(SBSymbolContextRef instance);

LLDB_API void SBSymbolContextSetModule(SBSymbolContextRef instance,
                                       SBModuleRef module);

LLDB_API void SBSymbolContextSetCompileUnit(SBSymbolContextRef instance,
                                            SBCompileUnitRef compile_unit);

LLDB_API void SBSymbolContextSetFunction(SBSymbolContextRef instance,
                                         SBFunctionRef function);

LLDB_API void SBSymbolContextSetBlock(SBSymbolContextRef instance,
                                      SBBlockRef block);

LLDB_API void SBSymbolContextSetLineEntry(SBSymbolContextRef instance,
                                          SBLineEntryRef line_entry);

LLDB_API void SBSymbolContextSetSymbol(SBSymbolContextRef instance,
                                       SBSymbolRef symbol);

LLDB_API SBSymbolContextRef SBSymbolContextGetParentOfInlinedScope(
    SBSymbolContextRef instance, SBAddressRef curr_frame_pc,
    SBAddressRef parent_frame_addr);

LLDB_API bool SBSymbolContextGetDescription(SBSymbolContextRef instance,
                                            SBStreamRef description);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBSymbolContextBinding_h_
