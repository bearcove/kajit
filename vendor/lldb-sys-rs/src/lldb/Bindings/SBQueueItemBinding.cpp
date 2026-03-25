//===-- SBQueueItemBinding.cpp ----------------------------------*- C++ -*-===//
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

SBQueueItemRef CreateSBQueueItem() {
  return reinterpret_cast<SBQueueItemRef>(new SBQueueItem());
}

SBQueueItemRef CloneSBQueueItem(SBQueueItemRef instance) {
  return reinterpret_cast<SBQueueItemRef>(
      new SBQueueItem(*reinterpret_cast<SBQueueItem *>(instance)));
}

void DisposeSBQueueItem(SBQueueItemRef instance) {
  delete reinterpret_cast<SBQueueItem *>(instance);
}

bool SBQueueItemIsValid(SBQueueItemRef instance) {
  SBQueueItem *unwrapped = reinterpret_cast<SBQueueItem *>(instance);
  return unwrapped->IsValid();
}

void SBQueueItemClear(SBQueueItemRef instance) {
  SBQueueItem *unwrapped = reinterpret_cast<SBQueueItem *>(instance);
  unwrapped->Clear();
}

enum lldb::QueueItemKind SBQueueItemGetKind(SBQueueItemRef instance) {
  SBQueueItem *unwrapped = reinterpret_cast<SBQueueItem *>(instance);
  return unwrapped->GetKind();
}

void SBQueueItemSetKind(SBQueueItemRef instance, lldb::QueueItemKind kind) {
  SBQueueItem *unwrapped = reinterpret_cast<SBQueueItem *>(instance);
  unwrapped->SetKind(kind);
}

SBAddressRef SBQueueItemGetAddress(SBQueueItemRef instance) {
  SBQueueItem *unwrapped = reinterpret_cast<SBQueueItem *>(instance);
  return reinterpret_cast<SBAddressRef>(new SBAddress(unwrapped->GetAddress()));
}

void SBQueueItemSetAddress(SBQueueItemRef instance, SBAddressRef addr) {
  SBQueueItem *unwrapped = reinterpret_cast<SBQueueItem *>(instance);
  unwrapped->SetAddress(*reinterpret_cast<SBAddress *>(addr));
}

SBThreadRef SBQueueItemGetExtendedBacktraceThread(SBQueueItemRef instance,
                                                  const char *type) {
  SBQueueItem *unwrapped = reinterpret_cast<SBQueueItem *>(instance);
  return reinterpret_cast<SBThreadRef>(
      new SBThread(unwrapped->GetExtendedBacktraceThread(type)));
}

#ifdef __cplusplus
}
#endif
