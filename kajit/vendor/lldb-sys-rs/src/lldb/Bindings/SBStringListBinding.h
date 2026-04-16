//===-- SBStringListBinding.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBStringListBinding_h_
#define LLDB_SBStringListBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBStringListRef CreateSBStringList(void);

LLDB_API SBStringListRef CloneSBStringList(SBStringListRef instance);

LLDB_API void DisposeSBStringList(SBStringListRef instance);

LLDB_API bool SBStringListIsValid(SBStringListRef instance);

LLDB_API void SBStringListAppendString(SBStringListRef instance,
                                       const char *str);

LLDB_API void SBStringListAppendList(SBStringListRef instance,
                                     const char **strv, int strc);

LLDB_API void SBStringListAppendList2(SBStringListRef instance,
                                      SBStringListRef strings);

LLDB_API uint32_t SBStringListGetSize(SBStringListRef instance);

LLDB_API const char *SBStringListGetStringAtIndex(SBStringListRef instance,
                                                  size_t idx);

LLDB_API void SBStringListClear(SBStringListRef instance);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBStringListBinding_h_
