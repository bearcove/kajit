//===-- SBThreadCollectionBinding.cpp ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/LLDB.h"
#include "lldb/API/SBThreadCollection.h"
#include "lldb/Bindings/LLDBBinding.h"

using namespace lldb;

#ifdef __cplusplus
extern "C" {
#endif

SBThreadCollectionRef CreateSBThreadCollection() {
  return reinterpret_cast<SBThreadCollectionRef>(new SBThreadCollection());
}

SBThreadCollectionRef CloneSBThreadCollection(SBThreadCollectionRef instance) {
  return reinterpret_cast<SBThreadCollectionRef>(new SBThreadCollection(
      *reinterpret_cast<SBThreadCollection *>(instance)));
}

void DisposeSBThreadCollection(SBThreadCollectionRef instance) {
  delete reinterpret_cast<SBThreadCollection *>(instance);
}

bool SBThreadCollectionIsValid(SBThreadCollectionRef instance) {
  SBThreadCollection *unwrapped =
      reinterpret_cast<SBThreadCollection *>(instance);
  return unwrapped->IsValid();
}

size_t SBThreadCollectionGetSize(SBThreadCollectionRef instance) {
  SBThreadCollection *unwrapped =
      reinterpret_cast<SBThreadCollection *>(instance);
  return unwrapped->GetSize();
}

SBThreadRef SBThreadCollectionGetThreadAtIndex(SBThreadCollectionRef instance,
                                               size_t idx) {
  SBThreadCollection *unwrapped =
      reinterpret_cast<SBThreadCollection *>(instance);
  return reinterpret_cast<SBThreadRef>(
      new SBThread(unwrapped->GetThreadAtIndex(idx)));
}

#ifdef __cplusplus
}
#endif
