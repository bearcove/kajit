//===-- SBEnvironmentBinding.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBEnvironmentBinding_h_
#define LLDB_SBEnvironmentBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBEnvironmentRef CreateSBEnvironment(void);

LLDB_API SBEnvironmentRef CloneSBEnvironment(SBEnvironmentRef instance);

LLDB_API void DisposeSBEnvironment(SBEnvironmentRef instance);

LLDB_API const char *SBEnvironmentGet(SBEnvironmentRef instance,
                                      const char *name);

LLDB_API size_t SBEnvironmentGetNumValues(SBEnvironmentRef instance);

LLDB_API const char *SBEnvironmentGetNameAtIndex(SBEnvironmentRef instance,
                                                 size_t index);

LLDB_API const char *SBEnvironmentGetValueAtIndex(SBEnvironmentRef instance,
                                                  size_t index);

LLDB_API SBStringListRef SBEnvironmentGetEntries(SBEnvironmentRef instance);

LLDB_API void SBEnvironmentPutEntry(SBEnvironmentRef instance,
                                    const char *name_and_value);

LLDB_API void SBEnvironmentSetEntries(SBEnvironmentRef instance,
                                      SBStringListRef entries, bool append);

LLDB_API bool SBEnvironmentSet(SBEnvironmentRef instance, const char *name,
                               const char *value, bool overwrite);

LLDB_API bool SBEnvironmentUnset(SBEnvironmentRef instance, const char *name);

LLDB_API void SBEnvironmentClear(SBEnvironmentRef instance);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBEnvironmentBinding_h_
