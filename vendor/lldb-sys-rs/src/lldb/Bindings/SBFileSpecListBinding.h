//===-- SBFileSpecListBinding.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBFileSpecListBinding_h_
#define LLDB_SBFileSpecListBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBFileSpecListRef CreateSBFileSpecList(void);

LLDB_API SBFileSpecListRef CloneSBFileSpecList(SBFileSpecListRef instance);

LLDB_API void DisposeSBFileSpecList(SBFileSpecListRef instance);

LLDB_API uint32_t SBFileSpecListGetSize(SBFileSpecListRef instance);

LLDB_API bool SBFileSpecListGetDescription(SBFileSpecListRef instance,
                                           SBStreamRef description);

LLDB_API void SBFileSpecListAppend(SBFileSpecListRef instance,
                                   SBFileSpecRef sb_file);

LLDB_API bool SBFileSpecListAppendIfUnique(SBFileSpecListRef instance,
                                           SBFileSpecRef sb_file);

LLDB_API void SBFileSpecListClear(SBFileSpecListRef instance);

LLDB_API uint32_t SBFileSpecListFindFileIndex(SBFileSpecListRef instance,
                                              uint32_t idx,
                                              SBFileSpecRef sb_file, bool full);

LLDB_API SBFileSpecRef
SBFileSpecListGetFileSpecAtIndex(SBFileSpecListRef instance, uint32_t idx);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBFileSpecListBinding_h_
