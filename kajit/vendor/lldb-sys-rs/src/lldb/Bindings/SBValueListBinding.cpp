//===-- SBValueListBinding.cpp ----------------------------------*- C++ -*-===//
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

SBValueListRef CreateSBValueList() {
  return reinterpret_cast<SBValueListRef>(new SBValueList());
}

SBValueListRef CloneSBValueList(SBValueListRef instance) {
  return reinterpret_cast<SBValueListRef>(
      new SBValueList(*reinterpret_cast<SBValueList *>(instance)));
}

void DisposeSBValueList(SBValueListRef instance) {
  delete reinterpret_cast<SBValueList *>(instance);
}

bool SBValueListIsValid(SBValueListRef instance) {
  SBValueList *unwrapped = reinterpret_cast<SBValueList *>(instance);
  return unwrapped->IsValid();
}

void SBValueListClear(SBValueListRef instance) {
  SBValueList *unwrapped = reinterpret_cast<SBValueList *>(instance);
  unwrapped->Clear();
}

void SBValueListAppend(SBValueListRef instance, SBValueRef val_obj) {
  SBValueList *unwrapped = reinterpret_cast<SBValueList *>(instance);
  unwrapped->Append(*reinterpret_cast<SBValue *>(val_obj));
}

void SBValueListAppendList(SBValueListRef instance, SBValueListRef value_list) {
  SBValueList *unwrapped = reinterpret_cast<SBValueList *>(instance);
  unwrapped->Append(*reinterpret_cast<SBValueList *>(value_list));
}

uint32_t SBValueListGetSize(SBValueListRef instance) {
  SBValueList *unwrapped = reinterpret_cast<SBValueList *>(instance);
  return unwrapped->GetSize();
}

SBValueRef SBValueListGetValueAtIndex(SBValueListRef instance, uint32_t idx) {
  SBValueList *unwrapped = reinterpret_cast<SBValueList *>(instance);
  return reinterpret_cast<SBValueRef>(
      new SBValue(unwrapped->GetValueAtIndex(idx)));
}

SBValueRef SBValueListGetFirstValueByName(SBValueListRef instance,
                                          const char *name) {
  SBValueList *unwrapped = reinterpret_cast<SBValueList *>(instance);
  return reinterpret_cast<SBValueRef>(
      new SBValue(unwrapped->GetFirstValueByName(name)));
}

SBValueRef SBValueListFindValueObjectByUID(SBValueListRef instance,
                                           lldb_user_id_t uid) {
  SBValueList *unwrapped = reinterpret_cast<SBValueList *>(instance);
  return reinterpret_cast<SBValueRef>(
      new SBValue(unwrapped->FindValueObjectByUID(uid)));
}

#ifdef __cplusplus
}
#endif
