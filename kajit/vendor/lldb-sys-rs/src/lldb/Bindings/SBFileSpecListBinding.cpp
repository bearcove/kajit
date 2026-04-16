//===-- SBFileSpecListBinding.cpp -------------------------------*- C++ -*-===//
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

SBFileSpecListRef CreateSBFileSpecList() {
  return reinterpret_cast<SBFileSpecListRef>(new SBFileSpecList());
}

SBFileSpecListRef CloneSBFileSpecList(SBFileSpecListRef instance) {
  return reinterpret_cast<SBFileSpecListRef>(
      new SBFileSpecList(*reinterpret_cast<SBFileSpecList *>(instance)));
}

void DisposeSBFileSpecList(SBFileSpecListRef instance) {
  delete reinterpret_cast<SBFileSpecList *>(instance);
}

uint32_t SBFileSpecListGetSize(SBFileSpecListRef instance) {
  SBFileSpecList *unwrapped = reinterpret_cast<SBFileSpecList *>(instance);
  return unwrapped->GetSize();
}

bool SBFileSpecListGetDescription(SBFileSpecListRef instance,
                                  SBStreamRef description) {
  SBFileSpecList *unwrapped = reinterpret_cast<SBFileSpecList *>(instance);
  return unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description));
}

void SBFileSpecListAppend(SBFileSpecListRef instance, SBFileSpecRef sb_file) {
  SBFileSpecList *unwrapped = reinterpret_cast<SBFileSpecList *>(instance);
  unwrapped->Append(*reinterpret_cast<SBFileSpec *>(sb_file));
}

bool SBFileSpecListAppendIfUnique(SBFileSpecListRef instance,
                                  SBFileSpecRef sb_file) {
  SBFileSpecList *unwrapped = reinterpret_cast<SBFileSpecList *>(instance);
  return unwrapped->AppendIfUnique(*reinterpret_cast<SBFileSpec *>(sb_file));
}

void SBFileSpecListClear(SBFileSpecListRef instance) {
  SBFileSpecList *unwrapped = reinterpret_cast<SBFileSpecList *>(instance);
  unwrapped->Clear();
}

uint32_t SBFileSpecListFindFileIndex(SBFileSpecListRef instance, uint32_t idx,
                                     SBFileSpecRef sb_file, bool full) {
  SBFileSpecList *unwrapped = reinterpret_cast<SBFileSpecList *>(instance);
  return unwrapped->FindFileIndex(idx, *reinterpret_cast<SBFileSpec *>(sb_file),
                                  full);
}

SBFileSpecRef SBFileSpecListGetFileSpecAtIndex(SBFileSpecListRef instance,
                                               uint32_t idx) {
  SBFileSpecList *unwrapped = reinterpret_cast<SBFileSpecList *>(instance);
  return reinterpret_cast<SBFileSpecRef>(
      new SBFileSpec(unwrapped->GetFileSpecAtIndex(idx)));
}

#ifdef __cplusplus
}
#endif
