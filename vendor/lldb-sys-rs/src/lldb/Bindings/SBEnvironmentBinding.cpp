//===-- SBEnvironmentBinding.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/LLDB.h"
#include "lldb/Bindings/LLDBBinding.h"
#include <stdarg.h>

using namespace lldb;

#ifdef __cplusplus
extern "C" {
#endif

SBEnvironmentRef CreateSBEnvironment() {
  return reinterpret_cast<SBEnvironmentRef>(new SBEnvironment());
}

SBEnvironmentRef CloneSBEnvironment(SBEnvironmentRef instance) {
  return reinterpret_cast<SBEnvironmentRef>(
      new SBEnvironment(*reinterpret_cast<SBEnvironment *>(instance)));
}

void DisposeSBEnvironment(SBEnvironmentRef instance) {
  delete reinterpret_cast<SBEnvironment *>(instance);
}

const char *SBEnvironmentGet(SBEnvironmentRef instance, const char *name) {
  SBEnvironment *unwrapped = reinterpret_cast<SBEnvironment *>(instance);
  return unwrapped->Get(name);
}

size_t SBEnvironmentGetNumValues(SBEnvironmentRef instance) {
  SBEnvironment *unwrapped = reinterpret_cast<SBEnvironment *>(instance);
  return unwrapped->GetNumValues();
}

const char *SBEnvironmentGetNameAtIndex(SBEnvironmentRef instance,
                                        size_t index) {
  SBEnvironment *unwrapped = reinterpret_cast<SBEnvironment *>(instance);
  return unwrapped->GetNameAtIndex(index);
}

const char *SBEnvironmentGetValueAtIndex(SBEnvironmentRef instance,
                                         size_t index) {
  SBEnvironment *unwrapped = reinterpret_cast<SBEnvironment *>(instance);
  return unwrapped->GetValueAtIndex(index);
}

SBStringListRef SBEnvironmentGetEntries(SBEnvironmentRef instance) {
  SBEnvironment *unwrapped = reinterpret_cast<SBEnvironment *>(instance);
  return reinterpret_cast<SBStringListRef>(
      new SBStringList(unwrapped->GetEntries()));
}

void SBEnvironmentPutEntry(SBEnvironmentRef instance,
                           const char *name_and_value) {
  SBEnvironment *unwrapped = reinterpret_cast<SBEnvironment *>(instance);
  unwrapped->PutEntry(name_and_value);
}

void SBEnvironmentSetEntries(SBEnvironmentRef instance, SBStringListRef entries,
                             bool append) {
  SBEnvironment *unwrapped = reinterpret_cast<SBEnvironment *>(instance);
  unwrapped->SetEntries(*reinterpret_cast<SBStringList *>(entries), append);
}

bool SBEnvironmentSet(SBEnvironmentRef instance, const char *name,
                      const char *value, bool overwrite) {
  SBEnvironment *unwrapped = reinterpret_cast<SBEnvironment *>(instance);
  return unwrapped->Set(name, value, overwrite);
}

bool SBEnvironmentUnset(SBEnvironmentRef instance, const char *name) {
  SBEnvironment *unwrapped = reinterpret_cast<SBEnvironment *>(instance);
  return unwrapped->Unset(name);
}

void SBEnvironmentClear(SBEnvironmentRef instance) {
  SBEnvironment *unwrapped = reinterpret_cast<SBEnvironment *>(instance);
  unwrapped->Clear();
}

#ifdef __cplusplus
}
#endif
