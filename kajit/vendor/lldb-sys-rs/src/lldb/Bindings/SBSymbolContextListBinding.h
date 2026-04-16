//===-- SBSymbolContextListBinding.h ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBSymbolContextListBinding_h_
#define LLDB_SBSymbolContextListBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBSymbolContextListRef CreateSBSymbolContextList(void);

LLDB_API SBSymbolContextListRef
CloneSBSymbolContextList(SBSymbolContextListRef instance);

LLDB_API void DisposeSBSymbolContextList(SBSymbolContextListRef instance);

LLDB_API bool SBSymbolContextListIsValid(SBSymbolContextListRef instance);

LLDB_API uint32_t SBSymbolContextListGetSize(SBSymbolContextListRef instance);

LLDB_API SBSymbolContextRef SBSymbolContextListGetContextAtIndex(
    SBSymbolContextListRef instance, uint32_t idx);

LLDB_API bool SBSymbolContextListGetDescription(SBSymbolContextListRef instance,
                                                SBStreamRef description);

LLDB_API void SBSymbolContextListAppend(SBSymbolContextListRef instance,
                                        SBSymbolContextRef sc);

LLDB_API void SBSymbolContextListAppendList(SBSymbolContextListRef instance,
                                            SBSymbolContextListRef sc_list);

LLDB_API void SBSymbolContextListClear(SBSymbolContextListRef instance);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBSymbolContextListBinding_h_
