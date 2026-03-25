//===-- SBAttachInfoBinding.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBAttachInfoBinding_h_
#define LLDB_SBAttachInfoBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBAttachInfoRef CreateSBAttachInfo(void);

LLDB_API SBAttachInfoRef CreateSBAttachInfo2(lldb_pid_t pid);

LLDB_API SBAttachInfoRef CreateSBAttachInfo3(const char *path, bool wait_for);

LLDB_API SBAttachInfoRef CreateSBAttachInfo4(const char *path, bool wait_for,
                                             bool async);

LLDB_API SBAttachInfoRef CloneSBAttachInfo(SBAttachInfoRef instance);

LLDB_API void DisposeSBAttachInfo(SBAttachInfoRef instance);

LLDB_API lldb_pid_t SBAttachInfoGetProcessID(SBAttachInfoRef instance);

LLDB_API void SBAttachInfoSetProcessID(SBAttachInfoRef instance,
                                       lldb_pid_t pid);

LLDB_API void SBAttachInfoSetExecutable(SBAttachInfoRef instance,
                                        const char *path);

LLDB_API void SBAttachInfoSetExecutable2(SBAttachInfoRef instance,
                                         SBFileSpecRef exe_file);

LLDB_API bool SBAttachInfoGetWaitForLaunch(SBAttachInfoRef instance);

LLDB_API void SBAttachInfoSetWaitForLaunch(SBAttachInfoRef instance, bool b);

LLDB_API void SBAttachInfoSetWaitForLaunch2(SBAttachInfoRef instance, bool b,
                                            bool async);

LLDB_API bool SBAttachInfoGetIgnoreExisting(SBAttachInfoRef instance);

LLDB_API void SBAttachInfoSetIgnoreExisting(SBAttachInfoRef instance, bool b);

LLDB_API uint32_t SBAttachInfoGetResumeCount(SBAttachInfoRef instance);

LLDB_API void SBAttachInfoSetResumeCount(SBAttachInfoRef instance, uint32_t c);

LLDB_API const char *SBAttachInfoGetProcessPluginName(SBAttachInfoRef instance);

LLDB_API void SBAttachInfoSetProcessPluginName(SBAttachInfoRef instance,
                                               const char *plugin_name);

LLDB_API uint32_t SBAttachInfoGetUserID(SBAttachInfoRef instance);

LLDB_API uint32_t SBAttachInfoGetGroupID(SBAttachInfoRef instance);

LLDB_API bool SBAttachInfoUserIDIsValid(SBAttachInfoRef instance);

LLDB_API bool SBAttachInfoGroupIDIsValid(SBAttachInfoRef instance);

LLDB_API void SBAttachInfoSetUserID(SBAttachInfoRef instance, uint32_t uid);

LLDB_API void SBAttachInfoSetGroupID(SBAttachInfoRef instance, uint32_t gid);

LLDB_API uint32_t SBAttachInfoGetEffectiveUserID(SBAttachInfoRef instance);

LLDB_API uint32_t SBAttachInfoGetEffectiveGroupID(SBAttachInfoRef instance);

LLDB_API bool SBAttachInfoEffectiveUserIDIsValid(SBAttachInfoRef instance);

LLDB_API bool SBAttachInfoEffectiveGroupIDIsValid(SBAttachInfoRef instance);

LLDB_API void SBAttachInfoSetEffectiveUserID(SBAttachInfoRef instance,
                                             uint32_t uid);

LLDB_API void SBAttachInfoSetEffectiveGroupID(SBAttachInfoRef instance,
                                              uint32_t gid);

LLDB_API lldb_pid_t SBAttachInfoGetParentProcessID(SBAttachInfoRef instance);

LLDB_API void SBAttachInfoSetParentProcessID(SBAttachInfoRef instance,
                                             lldb_pid_t pid);

LLDB_API bool SBAttachInfoParentProcessIDIsValid(SBAttachInfoRef instance);

LLDB_API SBListenerRef SBAttachInfoGetListener(SBAttachInfoRef instance);

LLDB_API void SBAttachInfoSetListener(SBAttachInfoRef instance,
                                      SBListenerRef listener);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBAttachInfoBinding_h_
