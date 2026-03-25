//===-- SBStringListBinding.cpp ---------------------------------*- C++ -*-===//
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

SBStringListRef CreateSBStringList() {
  return reinterpret_cast<SBStringListRef>(new SBStringList());
}

SBStringListRef CloneSBStringList(SBStringListRef instance) {
  return reinterpret_cast<SBStringListRef>(
      new SBStringList(*reinterpret_cast<SBStringList *>(instance)));
}

void DisposeSBStringList(SBStringListRef instance) {
  delete reinterpret_cast<SBStringList *>(instance);
}

bool SBStringListIsValid(SBStringListRef instance) {
  SBStringList *unwrapped = reinterpret_cast<SBStringList *>(instance);
  return unwrapped->IsValid();
}

void SBStringListAppendString(SBStringListRef instance, const char *str) {
  SBStringList *unwrapped = reinterpret_cast<SBStringList *>(instance);
  unwrapped->AppendString(str);
}

void SBStringListAppendList(SBStringListRef instance, const char **strv,
                            int strc) {
  SBStringList *unwrapped = reinterpret_cast<SBStringList *>(instance);
  unwrapped->AppendList(strv, strc);
}

void SBStringListAppendList2(SBStringListRef instance,
                             SBStringListRef strings) {
  SBStringList *unwrapped = reinterpret_cast<SBStringList *>(instance);
  unwrapped->AppendList(*reinterpret_cast<SBStringList *>(strings));
}

uint32_t SBStringListGetSize(SBStringListRef instance) {
  SBStringList *unwrapped = reinterpret_cast<SBStringList *>(instance);
  return unwrapped->GetSize();
}

const char *SBStringListGetStringAtIndex(SBStringListRef instance, size_t idx) {
  SBStringList *unwrapped = reinterpret_cast<SBStringList *>(instance);
  return unwrapped->GetStringAtIndex(idx);
}

void SBStringListClear(SBStringListRef instance) {
  SBStringList *unwrapped = reinterpret_cast<SBStringList *>(instance);
  unwrapped->Clear();
}

#ifdef __cplusplus
}
#endif
