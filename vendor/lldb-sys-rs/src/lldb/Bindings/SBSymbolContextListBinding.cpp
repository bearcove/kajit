//===-- SBSymbolContextListBinding.cpp --------------------------*- C++ -*-===//
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

SBSymbolContextListRef CreateSBSymbolContextList() {
  return reinterpret_cast<SBSymbolContextListRef>(new SBSymbolContextList());
}

SBSymbolContextListRef
CloneSBSymbolContextList(SBSymbolContextListRef instance) {
  return reinterpret_cast<SBSymbolContextListRef>(new SBSymbolContextList(
      *reinterpret_cast<SBSymbolContextList *>(instance)));
}

void DisposeSBSymbolContextList(SBSymbolContextListRef instance) {
  delete reinterpret_cast<SBSymbolContextList *>(instance);
}

bool SBSymbolContextListIsValid(SBSymbolContextListRef instance) {
  SBSymbolContextList *unwrapped =
      reinterpret_cast<SBSymbolContextList *>(instance);
  return unwrapped->IsValid();
}

uint32_t SBSymbolContextListGetSize(SBSymbolContextListRef instance) {
  SBSymbolContextList *unwrapped =
      reinterpret_cast<SBSymbolContextList *>(instance);
  return unwrapped->GetSize();
}

SBSymbolContextRef
SBSymbolContextListGetContextAtIndex(SBSymbolContextListRef instance,
                                     uint32_t idx) {
  SBSymbolContextList *unwrapped =
      reinterpret_cast<SBSymbolContextList *>(instance);
  return reinterpret_cast<SBSymbolContextRef>(
      new SBSymbolContext(unwrapped->GetContextAtIndex(idx)));
}

bool SBSymbolContextListGetDescription(SBSymbolContextListRef instance,
                                       SBStreamRef description) {
  SBSymbolContextList *unwrapped =
      reinterpret_cast<SBSymbolContextList *>(instance);
  return unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description));
}

void SBSymbolContextListAppend(SBSymbolContextListRef instance,
                               SBSymbolContextRef sc) {
  SBSymbolContextList *unwrapped =
      reinterpret_cast<SBSymbolContextList *>(instance);
  unwrapped->Append(*reinterpret_cast<SBSymbolContext *>(sc));
}

void SBSymbolContextListAppendList(SBSymbolContextListRef instance,
                                   SBSymbolContextListRef sc_list) {
  SBSymbolContextList *unwrapped =
      reinterpret_cast<SBSymbolContextList *>(instance);
  unwrapped->Append(*reinterpret_cast<SBSymbolContextList *>(sc_list));
}

void SBSymbolContextListClear(SBSymbolContextListRef instance) {
  SBSymbolContextList *unwrapped =
      reinterpret_cast<SBSymbolContextList *>(instance);
  unwrapped->Clear();
}

#ifdef __cplusplus
}
#endif
