//===-- SBLaunchInfoBinding.cpp ---------------------------------*- C++ -*-===//
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

SBLaunchInfoRef CreateSBLaunchInfo(const char **argv) {
  return reinterpret_cast<SBLaunchInfoRef>(new SBLaunchInfo(argv));
}

SBLaunchInfoRef CloneSBLaunchInfo(SBLaunchInfoRef instance) {
  return reinterpret_cast<SBLaunchInfoRef>(
      new SBLaunchInfo(*reinterpret_cast<SBLaunchInfo *>(instance)));
}

void DisposeSBLaunchInfo(SBLaunchInfoRef instance) {
  delete reinterpret_cast<SBLaunchInfo *>(instance);
}

lldb_pid_t SBLaunchInfoGetProcessID(SBLaunchInfoRef instance) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  return unwrapped->GetProcessID();
}

uint32_t SBLaunchInfoGetUserID(SBLaunchInfoRef instance) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  return unwrapped->GetUserID();
}

uint32_t SBLaunchInfoGetGroupID(SBLaunchInfoRef instance) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  return unwrapped->GetGroupID();
}

bool SBLaunchInfoUserIDIsValid(SBLaunchInfoRef instance) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  return unwrapped->UserIDIsValid();
}

bool SBLaunchInfoGroupIDIsValid(SBLaunchInfoRef instance) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  return unwrapped->GroupIDIsValid();
}

void SBLaunchInfoSetUserID(SBLaunchInfoRef instance, uint32_t uid) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  unwrapped->SetUserID(uid);
}

void SBLaunchInfoSetGroupID(SBLaunchInfoRef instance, uint32_t gid) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  unwrapped->SetGroupID(gid);
}

SBFileSpecRef SBLaunchInfoGetExecutableFile(SBLaunchInfoRef instance) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  return reinterpret_cast<SBFileSpecRef>(
      new SBFileSpec(unwrapped->GetExecutableFile()));
}

void SBLaunchInfoSetExecutableFile(SBLaunchInfoRef instance,
                                   SBFileSpecRef exe_file,
                                   bool add_as_first_arg) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  unwrapped->SetExecutableFile(*reinterpret_cast<SBFileSpec *>(exe_file),
                               add_as_first_arg);
}

SBListenerRef SBLaunchInfoGetListener(SBLaunchInfoRef instance) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  return reinterpret_cast<SBListenerRef>(
      new SBListener(unwrapped->GetListener()));
}

void SBLaunchInfoSetListener(SBLaunchInfoRef instance, SBListenerRef listener) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  unwrapped->SetListener(*reinterpret_cast<SBListener *>(listener));
}

uint32_t SBLaunchInfoGetNumArguments(SBLaunchInfoRef instance) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  return unwrapped->GetNumArguments();
}

const char *SBLaunchInfoGetArgumentAtIndex(SBLaunchInfoRef instance,
                                           uint32_t idx) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  return unwrapped->GetArgumentAtIndex(idx);
}

void SBLaunchInfoSetArguments(SBLaunchInfoRef instance, const char **argv,
                              bool append) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  unwrapped->SetArguments(argv, append);
}

uint32_t SBLaunchInfoGetNumEnvironmentEntries(SBLaunchInfoRef instance) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  return unwrapped->GetNumEnvironmentEntries();
}

const char *SBLaunchInfoGetEnvironmentEntryAtIndex(SBLaunchInfoRef instance,
                                                   uint32_t idx) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  return unwrapped->GetEnvironmentEntryAtIndex(idx);
}

void SBLaunchInfoSetEnvironmentEntries(SBLaunchInfoRef instance,
                                       const char **envp, bool append) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  unwrapped->SetEnvironmentEntries(envp, append);
}

void SBLaunchInfoSetEnvironment(SBLaunchInfoRef instance,
                                SBEnvironmentRef environment, bool append) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  unwrapped->SetEnvironment(*reinterpret_cast<SBEnvironment *>(environment),
                            append);
}

SBEnvironmentRef SBLaunchInfoGetEnvironment(SBLaunchInfoRef instance) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  return reinterpret_cast<SBEnvironmentRef>(
      new SBEnvironment(unwrapped->GetEnvironment()));
}

void SBLaunchInfoClear(SBLaunchInfoRef instance) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  unwrapped->Clear();
}

const char *SBLaunchInfoGetWorkingDirectory(SBLaunchInfoRef instance) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  return unwrapped->GetWorkingDirectory();
}

void SBLaunchInfoSetWorkingDirectory(SBLaunchInfoRef instance,
                                     const char *working_dir) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  unwrapped->SetWorkingDirectory(working_dir);
}

uint32_t SBLaunchInfoGetLaunchFlags(SBLaunchInfoRef instance) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  return unwrapped->GetLaunchFlags();
}

void SBLaunchInfoSetLaunchFlags(SBLaunchInfoRef instance, uint32_t flags) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  unwrapped->SetLaunchFlags(flags);
}

const char *SBLaunchInfoGetProcessPluginName(SBLaunchInfoRef instance) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  return unwrapped->GetProcessPluginName();
}

void SBLaunchInfoSetProcessPluginName(SBLaunchInfoRef instance,
                                      const char *plugin_name) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  unwrapped->SetProcessPluginName(plugin_name);
}

const char *SBLaunchInfoGetShell(SBLaunchInfoRef instance) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  return unwrapped->GetShell();
}

void SBLaunchInfoSetShell(SBLaunchInfoRef instance, const char *path) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  unwrapped->SetShell(path);
}

bool SBLaunchInfoGetShellExpandArguments(SBLaunchInfoRef instance) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  return unwrapped->GetShellExpandArguments();
}

void SBLaunchInfoSetShellExpandArguments(SBLaunchInfoRef instance, bool glob) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  unwrapped->SetShellExpandArguments(glob);
}

uint32_t SBLaunchInfoGetResumeCount(SBLaunchInfoRef instance) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  return unwrapped->GetResumeCount();
}

void SBLaunchInfoSetResumeCount(SBLaunchInfoRef instance, uint32_t c) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  unwrapped->SetResumeCount(c);
}

bool SBLaunchInfoAddCloseFileAction(SBLaunchInfoRef instance, int fd) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  return unwrapped->AddCloseFileAction(fd);
}

bool SBLaunchInfoAddDuplicateFileAction(SBLaunchInfoRef instance, int fd,
                                        int dup_fd) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  return unwrapped->AddDuplicateFileAction(fd, dup_fd);
}

bool SBLaunchInfoAddOpenFileAction(SBLaunchInfoRef instance, int fd,
                                   const char *path, bool read, bool write) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  return unwrapped->AddOpenFileAction(fd, path, read, write);
}

bool SBLaunchInfoAddSuppressFileAction(SBLaunchInfoRef instance, int fd,
                                       bool read, bool write) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  return unwrapped->AddSuppressFileAction(fd, read, write);
}

void SBLaunchInfoSetLaunchEventData(SBLaunchInfoRef instance,
                                    const char *data) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  unwrapped->SetLaunchEventData(data);
}

const char *SBLaunchInfoGetLaunchEventData(SBLaunchInfoRef instance) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  return unwrapped->GetLaunchEventData();
}

bool SBLaunchInfoGetDetachOnError(SBLaunchInfoRef instance) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  return unwrapped->GetDetachOnError();
}

void SBLaunchInfoSetDetachOnError(SBLaunchInfoRef instance, bool enable) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  unwrapped->SetDetachOnError(enable);
}

const char *SBLaunchInfoGetScriptedProcessClassName(SBLaunchInfoRef instance) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  return unwrapped->GetScriptedProcessClassName();
}

void SBLaunchInfoSetScriptedProcessClassName(SBLaunchInfoRef instance,
                                             const char *class_name) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  unwrapped->SetScriptedProcessClassName(class_name);
}

SBStructuredDataRef
SBLaunchInfoGetScriptedProcessDictionary(SBLaunchInfoRef instance) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  return reinterpret_cast<SBStructuredDataRef>(
      new SBStructuredData(unwrapped->GetScriptedProcessDictionary()));
}

void SBLaunchInfoSetScriptedProcessDictionary(SBLaunchInfoRef instance,
                                              SBStructuredDataRef dict) {
  SBLaunchInfo *unwrapped = reinterpret_cast<SBLaunchInfo *>(instance);
  unwrapped->SetScriptedProcessDictionary(
      *reinterpret_cast<SBStructuredData *>(dict));
}

#ifdef __cplusplus
}
#endif
