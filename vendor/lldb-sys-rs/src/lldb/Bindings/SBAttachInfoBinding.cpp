//===-- SBAttachInfoBinding.cpp ---------------------------------*- C++ -*-===//
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

SBAttachInfoRef CreateSBAttachInfo() {
  return reinterpret_cast<SBAttachInfoRef>(new SBAttachInfo());
}

SBAttachInfoRef CreateSBAttachInfo2(lldb_pid_t pid) {
  return reinterpret_cast<SBAttachInfoRef>(new SBAttachInfo(pid));
}

SBAttachInfoRef CreateSBAttachInfo3(const char *path, bool wait_for) {
  return reinterpret_cast<SBAttachInfoRef>(new SBAttachInfo(path, wait_for));
}

SBAttachInfoRef CreateSBAttachInfo4(const char *path, bool wait_for,
                                    bool async) {
  return reinterpret_cast<SBAttachInfoRef>(
      new SBAttachInfo(path, wait_for, async));
}

SBAttachInfoRef CloneSBAttachInfo(SBAttachInfoRef instance) {
  return reinterpret_cast<SBAttachInfoRef>(
      new SBAttachInfo(*reinterpret_cast<SBAttachInfo *>(instance)));
}

void DisposeSBAttachInfo(SBAttachInfoRef instance) {
  delete reinterpret_cast<SBAttachInfo *>(instance);
}

lldb_pid_t SBAttachInfoGetProcessID(SBAttachInfoRef instance) {
  SBAttachInfo *unwrapped = reinterpret_cast<SBAttachInfo *>(instance);
  return unwrapped->GetProcessID();
}

void SBAttachInfoSetProcessID(SBAttachInfoRef instance, lldb_pid_t pid) {
  SBAttachInfo *unwrapped = reinterpret_cast<SBAttachInfo *>(instance);
  unwrapped->SetProcessID(pid);
}

void SBAttachInfoSetExecutable(SBAttachInfoRef instance, const char *path) {
  SBAttachInfo *unwrapped = reinterpret_cast<SBAttachInfo *>(instance);
  unwrapped->SetExecutable(path);
}

void SBAttachInfoSetExecutable2(SBAttachInfoRef instance,
                                SBFileSpecRef exe_file) {
  SBAttachInfo *unwrapped = reinterpret_cast<SBAttachInfo *>(instance);
  unwrapped->SetExecutable(*reinterpret_cast<SBFileSpec *>(exe_file));
}

bool SBAttachInfoGetWaitForLaunch(SBAttachInfoRef instance) {
  SBAttachInfo *unwrapped = reinterpret_cast<SBAttachInfo *>(instance);
  return unwrapped->GetWaitForLaunch();
}

void SBAttachInfoSetWaitForLaunch(SBAttachInfoRef instance, bool b) {
  SBAttachInfo *unwrapped = reinterpret_cast<SBAttachInfo *>(instance);
  unwrapped->SetWaitForLaunch(b);
}

void SBAttachInfoSetWaitForLaunch2(SBAttachInfoRef instance, bool b,
                                   bool async) {
  SBAttachInfo *unwrapped = reinterpret_cast<SBAttachInfo *>(instance);
  unwrapped->SetWaitForLaunch(b, async);
}

bool SBAttachInfoGetIgnoreExisting(SBAttachInfoRef instance) {
  SBAttachInfo *unwrapped = reinterpret_cast<SBAttachInfo *>(instance);
  return unwrapped->GetIgnoreExisting();
}

void SBAttachInfoSetIgnoreExisting(SBAttachInfoRef instance, bool b) {
  SBAttachInfo *unwrapped = reinterpret_cast<SBAttachInfo *>(instance);
  unwrapped->SetIgnoreExisting(b);
}

uint32_t SBAttachInfoGetResumeCount(SBAttachInfoRef instance) {
  SBAttachInfo *unwrapped = reinterpret_cast<SBAttachInfo *>(instance);
  return unwrapped->GetResumeCount();
}

void SBAttachInfoSetResumeCount(SBAttachInfoRef instance, uint32_t c) {
  SBAttachInfo *unwrapped = reinterpret_cast<SBAttachInfo *>(instance);
  unwrapped->SetResumeCount(c);
}

const char *SBAttachInfoGetProcessPluginName(SBAttachInfoRef instance) {
  SBAttachInfo *unwrapped = reinterpret_cast<SBAttachInfo *>(instance);
  return unwrapped->GetProcessPluginName();
}

void SBAttachInfoSetProcessPluginName(SBAttachInfoRef instance,
                                      const char *plugin_name) {
  SBAttachInfo *unwrapped = reinterpret_cast<SBAttachInfo *>(instance);
  unwrapped->SetProcessPluginName(plugin_name);
}

uint32_t SBAttachInfoGetUserID(SBAttachInfoRef instance) {
  SBAttachInfo *unwrapped = reinterpret_cast<SBAttachInfo *>(instance);
  return unwrapped->GetUserID();
}

uint32_t SBAttachInfoGetGroupID(SBAttachInfoRef instance) {
  SBAttachInfo *unwrapped = reinterpret_cast<SBAttachInfo *>(instance);
  return unwrapped->GetGroupID();
}

bool SBAttachInfoUserIDIsValid(SBAttachInfoRef instance) {
  SBAttachInfo *unwrapped = reinterpret_cast<SBAttachInfo *>(instance);
  return unwrapped->UserIDIsValid();
}

bool SBAttachInfoGroupIDIsValid(SBAttachInfoRef instance) {
  SBAttachInfo *unwrapped = reinterpret_cast<SBAttachInfo *>(instance);
  return unwrapped->GroupIDIsValid();
}

void SBAttachInfoSetUserID(SBAttachInfoRef instance, uint32_t uid) {
  SBAttachInfo *unwrapped = reinterpret_cast<SBAttachInfo *>(instance);
  unwrapped->SetUserID(uid);
}

void SBAttachInfoSetGroupID(SBAttachInfoRef instance, uint32_t gid) {
  SBAttachInfo *unwrapped = reinterpret_cast<SBAttachInfo *>(instance);
  unwrapped->SetGroupID(gid);
}

uint32_t SBAttachInfoGetEffectiveUserID(SBAttachInfoRef instance) {
  SBAttachInfo *unwrapped = reinterpret_cast<SBAttachInfo *>(instance);
  return unwrapped->GetEffectiveUserID();
}

uint32_t SBAttachInfoGetEffectiveGroupID(SBAttachInfoRef instance) {
  SBAttachInfo *unwrapped = reinterpret_cast<SBAttachInfo *>(instance);
  return unwrapped->GetEffectiveGroupID();
}

bool SBAttachInfoEffectiveUserIDIsValid(SBAttachInfoRef instance) {
  SBAttachInfo *unwrapped = reinterpret_cast<SBAttachInfo *>(instance);
  return unwrapped->EffectiveUserIDIsValid();
}

bool SBAttachInfoEffectiveGroupIDIsValid(SBAttachInfoRef instance) {
  SBAttachInfo *unwrapped = reinterpret_cast<SBAttachInfo *>(instance);
  return unwrapped->EffectiveGroupIDIsValid();
}

void SBAttachInfoSetEffectiveUserID(SBAttachInfoRef instance, uint32_t uid) {
  SBAttachInfo *unwrapped = reinterpret_cast<SBAttachInfo *>(instance);
  unwrapped->SetEffectiveUserID(uid);
}

void SBAttachInfoSetEffectiveGroupID(SBAttachInfoRef instance, uint32_t gid) {
  SBAttachInfo *unwrapped = reinterpret_cast<SBAttachInfo *>(instance);
  unwrapped->SetEffectiveGroupID(gid);
}

lldb_pid_t SBAttachInfoGetParentProcessID(SBAttachInfoRef instance) {
  SBAttachInfo *unwrapped = reinterpret_cast<SBAttachInfo *>(instance);
  return unwrapped->GetParentProcessID();
}

void SBAttachInfoSetParentProcessID(SBAttachInfoRef instance, lldb_pid_t pid) {
  SBAttachInfo *unwrapped = reinterpret_cast<SBAttachInfo *>(instance);
  unwrapped->SetParentProcessID(pid);
}

bool SBAttachInfoParentProcessIDIsValid(SBAttachInfoRef instance) {
  SBAttachInfo *unwrapped = reinterpret_cast<SBAttachInfo *>(instance);
  return unwrapped->ParentProcessIDIsValid();
}

SBListenerRef SBAttachInfoGetListener(SBAttachInfoRef instance) {
  SBAttachInfo *unwrapped = reinterpret_cast<SBAttachInfo *>(instance);
  return reinterpret_cast<SBListenerRef>(
      new SBListener(unwrapped->GetListener()));
}

void SBAttachInfoSetListener(SBAttachInfoRef instance, SBListenerRef listener) {
  SBAttachInfo *unwrapped = reinterpret_cast<SBAttachInfo *>(instance);
  unwrapped->SetListener(*reinterpret_cast<SBListener *>(listener));
}

#ifdef __cplusplus
}
#endif
