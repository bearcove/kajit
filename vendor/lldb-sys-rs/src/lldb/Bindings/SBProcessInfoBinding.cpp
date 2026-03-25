//===-- SBProcessInfoBinding.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/LLDB.h"
#include "lldb/API/SBThreadCollection.h"
#include "lldb/API/SBUnixSignals.h"
#include "lldb/Bindings/LLDBBinding.h"

using namespace lldb;

#ifdef __cplusplus
extern "C" {
#endif

SBProcessInfoRef CreateSBProcessInfo() {
  return reinterpret_cast<SBProcessInfoRef>(new SBProcessInfo());
}

SBProcessInfoRef CloneSBProcessInfo(SBProcessInfoRef instance) {
  return reinterpret_cast<SBProcessInfoRef>(
      new SBProcessInfo(*reinterpret_cast<SBProcessInfo *>(instance)));
}

void DisposeSBProcessInfo(SBProcessInfoRef instance) {
  delete reinterpret_cast<SBProcessInfo *>(instance);
}

bool SBProcessInfoIsValid(SBProcessInfoRef instance) {
  SBProcessInfo *unwrapped = reinterpret_cast<SBProcessInfo *>(instance);
  return unwrapped->IsValid();
}

const char *SBProcessInfoGetName(SBProcessInfoRef instance) {
  SBProcessInfo *unwrapped = reinterpret_cast<SBProcessInfo *>(instance);
  return unwrapped->GetName();
}

SBFileSpecRef SBProcessInfoGetExecutableFile(SBProcessInfoRef instance) {
  SBProcessInfo *unwrapped = reinterpret_cast<SBProcessInfo *>(instance);
  return reinterpret_cast<SBFileSpecRef>(
      new SBFileSpec(unwrapped->GetExecutableFile()));
}

lldb_pid_t SBProcessInfoGetProcessID(SBProcessInfoRef instance) {
  SBProcessInfo *unwrapped = reinterpret_cast<SBProcessInfo *>(instance);
  return unwrapped->GetProcessID();
}

uint32_t SBProcessInfoGetUserID(SBProcessInfoRef instance) {
  SBProcessInfo *unwrapped = reinterpret_cast<SBProcessInfo *>(instance);
  return unwrapped->GetUserID();
}

uint32_t SBProcessInfoGetGroupID(SBProcessInfoRef instance) {
  SBProcessInfo *unwrapped = reinterpret_cast<SBProcessInfo *>(instance);
  return unwrapped->GetGroupID();
}

bool SBProcessInfoUserIDIsValid(SBProcessInfoRef instance) {
  SBProcessInfo *unwrapped = reinterpret_cast<SBProcessInfo *>(instance);
  return unwrapped->UserIDIsValid();
}

bool SBProcessInfoGroupIDIsValid(SBProcessInfoRef instance) {
  SBProcessInfo *unwrapped = reinterpret_cast<SBProcessInfo *>(instance);
  return unwrapped->GroupIDIsValid();
}

uint32_t SBProcessInfoGetEffectiveUserID(SBProcessInfoRef instance) {
  SBProcessInfo *unwrapped = reinterpret_cast<SBProcessInfo *>(instance);
  return unwrapped->GetEffectiveUserID();
}

uint32_t SBProcessInfoGetEffectiveGroupID(SBProcessInfoRef instance) {
  SBProcessInfo *unwrapped = reinterpret_cast<SBProcessInfo *>(instance);
  return unwrapped->GetEffectiveGroupID();
}

bool SBProcessInfoEffectiveUserIDIsValid(SBProcessInfoRef instance) {
  SBProcessInfo *unwrapped = reinterpret_cast<SBProcessInfo *>(instance);
  return unwrapped->EffectiveUserIDIsValid();
}

bool SBProcessInfoEffectiveGroupIDIsValid(SBProcessInfoRef instance) {
  SBProcessInfo *unwrapped = reinterpret_cast<SBProcessInfo *>(instance);
  return unwrapped->EffectiveGroupIDIsValid();
}

lldb_pid_t SBProcessInfoGetParentProcessID(SBProcessInfoRef instance) {
  SBProcessInfo *unwrapped = reinterpret_cast<SBProcessInfo *>(instance);
  return unwrapped->GetParentProcessID();
}

const char *SBProcessInfoGetTriple(SBProcessInfoRef instance) {
  SBProcessInfo *unwrapped = reinterpret_cast<SBProcessInfo *>(instance);
  return unwrapped->GetTriple();
}

#ifdef __cplusplus
}
#endif
