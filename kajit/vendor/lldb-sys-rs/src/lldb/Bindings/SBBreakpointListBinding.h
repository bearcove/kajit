//===-- SBBreakpointListBinding.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBBreakpointListBinding_h_
#define LLDB_SBBreakpointListBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBBreakpointListRef CreateSBBreakpointList(SBTargetRef target);

LLDB_API SBBreakpointListRef
CloneSBBreakpointList(SBBreakpointListRef instance);

LLDB_API void DisposeSBBreakpointList(SBBreakpointListRef instance);

LLDB_API size_t SBBreakpointListGetSize(SBBreakpointListRef instance);

LLDB_API SBBreakpointRef
SBBreakpointListGetBreakpointAtIndex(SBBreakpointListRef instance, size_t idx);

LLDB_API SBBreakpointRef
SBBreakpointListFindBreakpointByID(SBBreakpointListRef instance, int break_id);

LLDB_API void SBBreakpointListAppend(SBBreakpointListRef instance,
                                     SBBreakpointRef sb_bkpt);

LLDB_API bool SBBreakpointListAppendIfUnique(SBBreakpointListRef instance,
                                             SBBreakpointRef sb_bkpt);

LLDB_API void SBBreakpointListAppendByID(SBBreakpointListRef instance,
                                         int break_id);

LLDB_API void SBBreakpointListClear(SBBreakpointListRef instance);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBBreakpointListBinding_h_
