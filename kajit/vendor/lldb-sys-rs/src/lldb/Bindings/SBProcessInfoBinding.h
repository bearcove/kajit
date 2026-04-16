//===-- SBProcessInfoBinding.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBProcessInfoBinding_h_
#define LLDB_SBProcessInfoBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBProcessInfoRef CreateSBProcessInfo(void);

LLDB_API SBProcessInfoRef CloneSBProcessInfo(SBProcessInfoRef instance);

LLDB_API void DisposeSBProcessInfo(SBProcessInfoRef instance);

LLDB_API bool SBProcessInfoIsValid(SBProcessInfoRef instance);

LLDB_API const char *SBProcessInfoGetName(SBProcessInfoRef instance);

LLDB_API SBFileSpecRef
SBProcessInfoGetExecutableFile(SBProcessInfoRef instance);

LLDB_API lldb_pid_t SBProcessInfoGetProcessID(SBProcessInfoRef instance);

LLDB_API uint32_t SBProcessInfoGetUserID(SBProcessInfoRef instance);

LLDB_API uint32_t SBProcessInfoGetGroupID(SBProcessInfoRef instance);

LLDB_API bool SBProcessInfoUserIDIsValid(SBProcessInfoRef instance);

LLDB_API bool SBProcessInfoGroupIDIsValid(SBProcessInfoRef instance);

LLDB_API uint32_t SBProcessInfoGetEffectiveUserID(SBProcessInfoRef instance);

LLDB_API uint32_t SBProcessInfoGetEffectiveGroupID(SBProcessInfoRef instance);

LLDB_API bool SBProcessInfoEffectiveUserIDIsValid(SBProcessInfoRef instance);

LLDB_API bool SBProcessInfoEffectiveGroupIDIsValid(SBProcessInfoRef instance);

LLDB_API lldb_pid_t SBProcessInfoGetParentProcessID(SBProcessInfoRef instance);

LLDB_API const char *SBProcessInfoGetTriple(SBProcessInfoRef instance);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBProcessInfoBinding_h_
