//===-- SBPlatformBinding.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBPlatformBinding_h_
#define LLDB_SBPlatformBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBPlatformConnectOptionsRef
CreateSBPlatformConnectOptions(const char *url);

LLDB_API SBPlatformConnectOptionsRef
CloneSBPlatformConnectOptions(SBPlatformConnectOptionsRef instance);

LLDB_API void
DisposeSBPlatformConnectOptions(SBPlatformConnectOptionsRef instance);

LLDB_API const char *
SBPlatformConnectOptionsGetURL(SBPlatformConnectOptionsRef instance);

LLDB_API void
SBPlatformConnectOptionsSetURL(SBPlatformConnectOptionsRef instance,
                               const char *url);

LLDB_API bool
SBPlatformConnectOptionsGetRsyncEnabled(SBPlatformConnectOptionsRef instance);

LLDB_API void SBPlatformConnectOptionsEnableRsync(
    SBPlatformConnectOptionsRef instance, const char *options,
    const char *remote_path_prefix, bool omit_remote_hostname);

LLDB_API void
SBPlatformConnectOptionsDisableRsync(SBPlatformConnectOptionsRef instance);

LLDB_API const char *SBPlatformConnectOptionsGetLocalCacheDirectory(
    SBPlatformConnectOptionsRef instance);

LLDB_API void SBPlatformConnectOptionsSetLocalCacheDirectory(
    SBPlatformConnectOptionsRef instance, const char *path);

LLDB_API SBPlatformShellCommandRef
CreateSBPlatformShellCommand(const char *shell_command);

LLDB_API SBPlatformShellCommandRef
CreateSBPlatformShellCommand2(const char *shell, const char *shell_command);

LLDB_API SBPlatformShellCommandRef
CloneSBPlatformShellCommand(SBPlatformShellCommandRef instance);

LLDB_API void DisposeSBPlatformShellCommand(SBPlatformShellCommandRef instance);

LLDB_API void SBPlatformShellCommandClear(SBPlatformShellCommandRef instance);

LLDB_API const char *
SBPlatformShellCommandGetShell(SBPlatformShellCommandRef instance);

LLDB_API void SBPlatformShellCommandSetShel(SBPlatformShellCommandRef instance,
                                            const char *shell);

LLDB_API const char *
SBPlatformShellCommandGetCommand(SBPlatformShellCommandRef instance);

LLDB_API void
SBPlatformShellCommandSetCommand(SBPlatformShellCommandRef instance,
                                 const char *shell_command);

LLDB_API const char *
SBPlatformShellCommandGetWorkingDirectory(SBPlatformShellCommandRef instance);

LLDB_API void
SBPlatformShellCommandSetWorkingDirectory(SBPlatformShellCommandRef instance,
                                          const char *path);

LLDB_API uint32_t
SBPlatformShellCommandGetTimeoutSeconds(SBPlatformShellCommandRef instance);

LLDB_API void
SBPlatformShellCommandSetTimeoutSeconds(SBPlatformShellCommandRef instance,
                                        uint32_t sec);

LLDB_API int
SBPlatformShellCommandGetSignal(SBPlatformShellCommandRef instance);

LLDB_API int
SBPlatformShellCommandGetStatus(SBPlatformShellCommandRef instance);

LLDB_API const char *
SBPlatformShellCommandGetOutput(SBPlatformShellCommandRef instance);

LLDB_API SBPlatformRef CreateSBPlatform(void);

LLDB_API SBPlatformRef CreateSBPlatform2(const char *platform_name);

LLDB_API SBPlatformRef CloneSBPlatform(SBPlatformRef instance);

LLDB_API void DisposeSBPlatform(SBPlatformRef instance);

LLDB_API SBPlatformRef SBPlatformGetHostPlatform(void);

LLDB_API bool SBPlatformIsValid(SBPlatformRef instance);

LLDB_API void SBPlatformClear(SBPlatformRef instance);

LLDB_API const char *SBPlatformGetWorkingDirectory(SBPlatformRef instance);

LLDB_API bool SBPlatformSetWorkingDirectory(SBPlatformRef instance,
                                            const char *path);

LLDB_API const char *SBPlatformGetName(SBPlatformRef instance);

LLDB_API SBErrorRef SBPlatformConnectRemote(
    SBPlatformRef instance, SBPlatformConnectOptionsRef connect_options);

LLDB_API void SBPlatformDisconnectRemote(SBPlatformRef instance);

LLDB_API bool SBPlatformIsConnected(SBPlatformRef instance);

LLDB_API const char *SBPlatformGetTriple(SBPlatformRef instance);

LLDB_API const char *SBPlatformGetHostname(SBPlatformRef instance);

LLDB_API const char *SBPlatformGetOSBuild(SBPlatformRef instance);

LLDB_API const char *SBPlatformGetOSDescription(SBPlatformRef instance);

LLDB_API uint32_t SBPlatformGetOSMajorVersion(SBPlatformRef instance);

LLDB_API uint32_t SBPlatformGetOSMinorVersion(SBPlatformRef instance);

LLDB_API uint32_t SBPlatformGetOSUpdateVersion(SBPlatformRef instance);

LLDB_API void SBPlatformSetSDKRoot(SBPlatformRef instance, const char *sysroot);

LLDB_API SBErrorRef SBPlatformPut(SBPlatformRef instance, SBFileSpecRef src,
                                  SBFileSpecRef dst);

LLDB_API SBErrorRef SBPlatformGet(SBPlatformRef instance, SBFileSpecRef src,
                                  SBFileSpecRef dst);

LLDB_API SBErrorRef SBPlatformInstall(SBPlatformRef instance, SBFileSpecRef src,
                                      SBFileSpecRef dst);

LLDB_API SBErrorRef SBPlatformRun(SBPlatformRef instance,
                                  SBPlatformShellCommandRef shell_command);

LLDB_API SBErrorRef SBPlatformLaunch(SBPlatformRef instance,
                                     SBLaunchInfoRef launch_info);

LLDB_API SBErrorRef SBPlatformKill(SBPlatformRef instance,
                                   const lldb_pid_t pid);

LLDB_API SBErrorRef SBPlatformMakeDirectory(SBPlatformRef instance,
                                            const char *path,
                                            uint32_t file_permissions);

LLDB_API uint32_t SBPlatformGetFilePermissions(SBPlatformRef instance,
                                               const char *path);

LLDB_API SBErrorRef SBPlatformSetFilePermissions(SBPlatformRef instance,
                                                 const char *path,
                                                 uint32_t file_permissions);

LLDB_API SBEnvironmentRef SBPlatformGetEnvironment(SBPlatformRef instance);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBPlatformBinding_h_
