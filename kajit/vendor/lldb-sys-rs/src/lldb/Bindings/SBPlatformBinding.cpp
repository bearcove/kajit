//===-- SBPlatformBinding.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/LLDB.h"
#include "lldb/Bindings/LLDBBinding.h"

using namespace lldb;

#ifdef __cplusplus
extern "C" {
#endif

SBPlatformConnectOptionsRef CreateSBPlatformConnectOptions(const char *url) {
  return reinterpret_cast<SBPlatformConnectOptionsRef>(
      new SBPlatformConnectOptions(url));
}

void DisposeSBPlatformConnectOptions(SBPlatformConnectOptionsRef instance) {
  delete reinterpret_cast<SBPlatformConnectOptions *>(instance);
}

SBPlatformConnectOptionsRef
CloneSBPlatformConnectOptions(SBPlatformConnectOptionsRef instance) {
  return reinterpret_cast<SBPlatformConnectOptionsRef>(
      new SBPlatformConnectOptions(
          *reinterpret_cast<SBPlatformConnectOptions *>(instance)));
}

const char *
SBPlatformConnectOptionsGetURL(SBPlatformConnectOptionsRef instance) {
  SBPlatformConnectOptions *unwrapped =
      reinterpret_cast<SBPlatformConnectOptions *>(instance);
  return unwrapped->GetURL();
}

void SBPlatformConnectOptionsSetURL(SBPlatformConnectOptionsRef instance,
                                    const char *url) {
  SBPlatformConnectOptions *unwrapped =
      reinterpret_cast<SBPlatformConnectOptions *>(instance);
  unwrapped->SetURL(url);
}

bool SBPlatformConnectOptionsGetRsyncEnabled(
    SBPlatformConnectOptionsRef instance) {
  SBPlatformConnectOptions *unwrapped =
      reinterpret_cast<SBPlatformConnectOptions *>(instance);
  return unwrapped->GetRsyncEnabled();
}

void SBPlatformConnectOptionsEnableRsync(SBPlatformConnectOptionsRef instance,
                                         const char *options,
                                         const char *remote_path_prefix,
                                         bool omit_remote_hostname) {
  SBPlatformConnectOptions *unwrapped =
      reinterpret_cast<SBPlatformConnectOptions *>(instance);
  unwrapped->EnableRsync(options, remote_path_prefix, omit_remote_hostname);
}

void SBPlatformConnectOptionsDisableRsync(
    SBPlatformConnectOptionsRef instance) {
  SBPlatformConnectOptions *unwrapped =
      reinterpret_cast<SBPlatformConnectOptions *>(instance);
  unwrapped->DisableRsync();
}

const char *SBPlatformConnectOptionsGetLocalCacheDirectory(
    SBPlatformConnectOptionsRef instance) {
  SBPlatformConnectOptions *unwrapped =
      reinterpret_cast<SBPlatformConnectOptions *>(instance);
  return unwrapped->GetLocalCacheDirectory();
}

void SBPlatformConnectOptionsSetLocalCacheDirectory(
    SBPlatformConnectOptionsRef instance, const char *path) {
  SBPlatformConnectOptions *unwrapped =
      reinterpret_cast<SBPlatformConnectOptions *>(instance);
  unwrapped->SetLocalCacheDirectory(path);
}

SBPlatformShellCommandRef
CreateSBPlatformShellCommand(const char *shell_command) {
  return reinterpret_cast<SBPlatformShellCommandRef>(
      new SBPlatformShellCommand(shell_command));
}

SBPlatformShellCommandRef
CreateSBPlatformShellCommand2(const char *shell, const char *shell_command) {
  return reinterpret_cast<SBPlatformShellCommandRef>(
      new SBPlatformShellCommand(shell, shell_command));
}

SBPlatformShellCommandRef
CloneSBPlatformShellCommand(SBPlatformShellCommandRef instance) {
  return reinterpret_cast<SBPlatformShellCommandRef>(new SBPlatformShellCommand(
      *reinterpret_cast<SBPlatformShellCommand *>(instance)));
}

void DisposeSBPlatformShellCommand(SBPlatformShellCommandRef instance) {
  delete reinterpret_cast<SBPlatformShellCommand *>(instance);
}

void SBPlatformShellCommandClear(SBPlatformShellCommandRef instance) {
  SBPlatformShellCommand *unwrapped =
      reinterpret_cast<SBPlatformShellCommand *>(instance);
  unwrapped->Clear();
}

const char *SBPlatformShellCommandGetShell(SBPlatformShellCommandRef instance) {
  SBPlatformShellCommand *unwrapped =
      reinterpret_cast<SBPlatformShellCommand *>(instance);
  return unwrapped->GetShell();
}

void SBPlatformShellCommandSetShell(SBPlatformShellCommandRef instance,
                                    const char *shell) {
  SBPlatformShellCommand *unwrapped =
      reinterpret_cast<SBPlatformShellCommand *>(instance);
  unwrapped->SetShell(shell);
}

const char *
SBPlatformShellCommandGetCommand(SBPlatformShellCommandRef instance) {
  SBPlatformShellCommand *unwrapped =
      reinterpret_cast<SBPlatformShellCommand *>(instance);
  return unwrapped->GetCommand();
}

void SBPlatformShellCommandSetCommand(SBPlatformShellCommandRef instance,
                                      const char *shell_command) {
  SBPlatformShellCommand *unwrapped =
      reinterpret_cast<SBPlatformShellCommand *>(instance);
  unwrapped->SetCommand(shell_command);
}

const char *
SBPlatformShellCommandGetWorkingDirectory(SBPlatformShellCommandRef instance) {
  SBPlatformShellCommand *unwrapped =
      reinterpret_cast<SBPlatformShellCommand *>(instance);
  return unwrapped->GetWorkingDirectory();
}

void SBPlatformShellCommandSetWorkingDirectory(
    SBPlatformShellCommandRef instance, const char *path) {
  SBPlatformShellCommand *unwrapped =
      reinterpret_cast<SBPlatformShellCommand *>(instance);
  unwrapped->SetWorkingDirectory(path);
}

uint32_t
SBPlatformShellCommandGetTimeoutSeconds(SBPlatformShellCommandRef instance) {
  SBPlatformShellCommand *unwrapped =
      reinterpret_cast<SBPlatformShellCommand *>(instance);
  return unwrapped->GetTimeoutSeconds();
}

void SBPlatformShellCommandSetTimeoutSeconds(SBPlatformShellCommandRef instance,
                                             uint32_t sec) {
  SBPlatformShellCommand *unwrapped =
      reinterpret_cast<SBPlatformShellCommand *>(instance);
  unwrapped->SetTimeoutSeconds(sec);
}

int SBPlatformShellCommandGetSignal(SBPlatformShellCommandRef instance) {
  SBPlatformShellCommand *unwrapped =
      reinterpret_cast<SBPlatformShellCommand *>(instance);
  return unwrapped->GetSignal();
}

int SBPlatformShellCommandGetStatus(SBPlatformShellCommandRef instance) {
  SBPlatformShellCommand *unwrapped =
      reinterpret_cast<SBPlatformShellCommand *>(instance);
  return unwrapped->GetStatus();
}

const char *
SBPlatformShellCommandGetOutput(SBPlatformShellCommandRef instance) {
  SBPlatformShellCommand *unwrapped =
      reinterpret_cast<SBPlatformShellCommand *>(instance);
  return unwrapped->GetOutput();
}

SBPlatformRef CreateSBPlatform() {
  return reinterpret_cast<SBPlatformRef>(new SBPlatform());
}

SBPlatformRef CreateSBPlatform2(const char *platform_name) {
  return reinterpret_cast<SBPlatformRef>(new SBPlatform(platform_name));
}

SBPlatformRef CloneSBPlatform(SBPlatformRef instance) {
  return reinterpret_cast<SBPlatformRef>(
      new SBPlatform(*reinterpret_cast<SBPlatform *>(instance)));
}

void DisposeSBPlatform(SBPlatformRef instance) {
  delete reinterpret_cast<SBPlatform *>(instance);
}

SBPlatformRef SBPlatformGetHostPlatform() {
  return reinterpret_cast<SBPlatformRef>(
      new SBPlatform(SBPlatform::GetHostPlatform()));
}

bool SBPlatformIsValid(SBPlatformRef instance) {
  SBPlatform *unwrapped = reinterpret_cast<SBPlatform *>(instance);
  return unwrapped->IsValid();
}

void SBPlatformClear(SBPlatformRef instance) {
  SBPlatform *unwrapped = reinterpret_cast<SBPlatform *>(instance);
  unwrapped->Clear();
}

const char *SBPlatformGetWorkingDirectory(SBPlatformRef instance) {
  SBPlatform *unwrapped = reinterpret_cast<SBPlatform *>(instance);
  return unwrapped->GetWorkingDirectory();
}

bool SBPlatformSetWorkingDirectory(SBPlatformRef instance, const char *path) {
  SBPlatform *unwrapped = reinterpret_cast<SBPlatform *>(instance);
  return unwrapped->SetWorkingDirectory(path);
}

const char *SBPlatformGetName(SBPlatformRef instance) {
  SBPlatform *unwrapped = reinterpret_cast<SBPlatform *>(instance);
  return unwrapped->GetName();
}

SBErrorRef
SBPlatformConnectRemote(SBPlatformRef instance,
                        SBPlatformConnectOptionsRef connect_options) {
  SBPlatform *unwrapped = reinterpret_cast<SBPlatform *>(instance);
  return reinterpret_cast<SBErrorRef>(new SBError(unwrapped->ConnectRemote(
      *reinterpret_cast<SBPlatformConnectOptions *>(connect_options))));
}

void SBPlatformDisconnectRemote(SBPlatformRef instance) {
  SBPlatform *unwrapped = reinterpret_cast<SBPlatform *>(instance);
  unwrapped->DisconnectRemote();
}

bool SBPlatformIsConnected(SBPlatformRef instance) {
  SBPlatform *unwrapped = reinterpret_cast<SBPlatform *>(instance);
  return unwrapped->IsConnected();
}

const char *SBPlatformGetTriple(SBPlatformRef instance) {
  SBPlatform *unwrapped = reinterpret_cast<SBPlatform *>(instance);
  return unwrapped->GetTriple();
}

const char *SBPlatformGetHostname(SBPlatformRef instance) {
  SBPlatform *unwrapped = reinterpret_cast<SBPlatform *>(instance);
  return unwrapped->GetHostname();
}

const char *SBPlatformGetOSBuild(SBPlatformRef instance) {
  SBPlatform *unwrapped = reinterpret_cast<SBPlatform *>(instance);
  return unwrapped->GetOSBuild();
}

const char *SBPlatformGetOSDescription(SBPlatformRef instance) {
  SBPlatform *unwrapped = reinterpret_cast<SBPlatform *>(instance);
  return unwrapped->GetOSDescription();
}

uint32_t SBPlatformGetOSMajorVersion(SBPlatformRef instance) {
  SBPlatform *unwrapped = reinterpret_cast<SBPlatform *>(instance);
  return unwrapped->GetOSMajorVersion();
}

uint32_t SBPlatformGetOSMinorVersion(SBPlatformRef instance) {
  SBPlatform *unwrapped = reinterpret_cast<SBPlatform *>(instance);
  return unwrapped->GetOSMinorVersion();
}

uint32_t SBPlatformGetOSUpdateVersion(SBPlatformRef instance) {
  SBPlatform *unwrapped = reinterpret_cast<SBPlatform *>(instance);
  return unwrapped->GetOSUpdateVersion();
}

void SBPlatformSetSDKRoot(SBPlatformRef instance, const char *sysroot) {
  SBPlatform *unwrapped = reinterpret_cast<SBPlatform *>(instance);
  unwrapped->SetSDKRoot(sysroot);
}

SBErrorRef SBPlatformPut(SBPlatformRef instance, SBFileSpecRef src,
                         SBFileSpecRef dst) {
  SBPlatform *unwrapped = reinterpret_cast<SBPlatform *>(instance);
  return reinterpret_cast<SBErrorRef>(
      new SBError(unwrapped->Put(*reinterpret_cast<SBFileSpec *>(src),
                                 *reinterpret_cast<SBFileSpec *>(dst))));
}

SBErrorRef SBPlatformGet(SBPlatformRef instance, SBFileSpecRef src,
                         SBFileSpecRef dst) {
  SBPlatform *unwrapped = reinterpret_cast<SBPlatform *>(instance);
  return reinterpret_cast<SBErrorRef>(
      new SBError(unwrapped->Get(*reinterpret_cast<SBFileSpec *>(src),
                                 *reinterpret_cast<SBFileSpec *>(dst))));
}

SBErrorRef SBPlatformInstall(SBPlatformRef instance, SBFileSpecRef src,
                             SBFileSpecRef dst) {
  SBPlatform *unwrapped = reinterpret_cast<SBPlatform *>(instance);
  return reinterpret_cast<SBErrorRef>(
      new SBError(unwrapped->Install(*reinterpret_cast<SBFileSpec *>(src),
                                     *reinterpret_cast<SBFileSpec *>(dst))));
}

SBErrorRef SBPlatformRun(SBPlatformRef instance,
                         SBPlatformShellCommandRef shell_command) {
  SBPlatform *unwrapped = reinterpret_cast<SBPlatform *>(instance);
  return reinterpret_cast<SBErrorRef>(new SBError(unwrapped->Run(
      *reinterpret_cast<SBPlatformShellCommand *>(shell_command))));
}

SBErrorRef SBPlatformLaunch(SBPlatformRef instance,
                            SBLaunchInfoRef launch_info) {
  SBPlatform *unwrapped = reinterpret_cast<SBPlatform *>(instance);
  return reinterpret_cast<SBErrorRef>(new SBError(
      unwrapped->Launch(*reinterpret_cast<SBLaunchInfo *>(launch_info))));
}

SBErrorRef SBPlatformKill(SBPlatformRef instance, const lldb_pid_t pid) {
  SBPlatform *unwrapped = reinterpret_cast<SBPlatform *>(instance);
  return reinterpret_cast<SBErrorRef>(new SBError(unwrapped->Kill(pid)));
}

SBErrorRef SBPlatformMakeDirectory(SBPlatformRef instance, const char *path,
                                   uint32_t file_permissions) {
  SBPlatform *unwrapped = reinterpret_cast<SBPlatform *>(instance);
  return reinterpret_cast<SBErrorRef>(
      new SBError(unwrapped->MakeDirectory(path, file_permissions)));
}

uint32_t SBPlatformGetFilePermissions(SBPlatformRef instance,
                                      const char *path) {
  SBPlatform *unwrapped = reinterpret_cast<SBPlatform *>(instance);
  return unwrapped->GetFilePermissions(path);
}

SBErrorRef SBPlatformSetFilePermissions(SBPlatformRef instance,
                                        const char *path,
                                        uint32_t file_permissions) {
  SBPlatform *unwrapped = reinterpret_cast<SBPlatform *>(instance);
  return reinterpret_cast<SBErrorRef>(
      new SBError(unwrapped->SetFilePermissions(path, file_permissions)));
}

SBEnvironmentRef SBPlatformGetEnvironment(SBPlatformRef instance) {
  SBPlatform *unwrapped = reinterpret_cast<SBPlatform *>(instance);
  return reinterpret_cast<SBEnvironmentRef>(
      new SBEnvironment(unwrapped->GetEnvironment()));
}

#ifdef __cplusplus
}
#endif
