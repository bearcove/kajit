//===-- SBEventBinding.cpp --------------------------------------*- C++ -*-===//
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

SBEventRef CreateSBEvent() {
  return reinterpret_cast<SBEventRef>(new SBEvent());
}

SBEventRef CreateSBEvent2(uint32_t event, const char *cstr, uint32_t cstr_len) {
  return reinterpret_cast<SBEventRef>(new SBEvent(event, cstr, cstr_len));
}

SBEventRef CloneSBEvent(SBEventRef instance) {
  return reinterpret_cast<SBEventRef>(
      new SBEvent(*reinterpret_cast<SBEvent *>(instance)));
}

void DisposeSBEvent(SBEventRef instance) {
  delete reinterpret_cast<SBEvent *>(instance);
}

bool SBEventIsValid(SBEventRef instance) {
  SBEvent *unwrapped = reinterpret_cast<SBEvent *>(instance);
  return unwrapped->IsValid();
}

const char *SBEventGetDataFlavor(SBEventRef instance) {
  SBEvent *unwrapped = reinterpret_cast<SBEvent *>(instance);
  return unwrapped->GetDataFlavor();
}

uint32_t SBEventGetType(SBEventRef instance) {
  SBEvent *unwrapped = reinterpret_cast<SBEvent *>(instance);
  return unwrapped->GetType();
}

SBBroadcasterRef SBEventGetBroadcaster(SBEventRef instance) {
  SBEvent *unwrapped = reinterpret_cast<SBEvent *>(instance);
  return reinterpret_cast<SBBroadcasterRef>(
      new SBBroadcaster(unwrapped->GetBroadcaster()));
}

const char *SBEventGetBroadcasterClass(SBEventRef instance) {
  SBEvent *unwrapped = reinterpret_cast<SBEvent *>(instance);
  return unwrapped->GetBroadcasterClass();
}

bool SBEventBroadcasterMatchesPtr(SBEventRef instance,
                                  SBBroadcasterRef broadcaster) {
  SBEvent *unwrapped = reinterpret_cast<SBEvent *>(instance);
  return unwrapped->BroadcasterMatchesPtr(
      reinterpret_cast<SBBroadcaster *>(broadcaster));
}

bool SBEventBroadcasterMatchesRef(SBEventRef instance,
                                  SBBroadcasterRef broadcaster) {
  SBEvent *unwrapped = reinterpret_cast<SBEvent *>(instance);
  return unwrapped->BroadcasterMatchesRef(
      *reinterpret_cast<SBBroadcaster *>(broadcaster));
}

void SBEventClear(SBEventRef instance) {
  SBEvent *unwrapped = reinterpret_cast<SBEvent *>(instance);
  unwrapped->Clear();
}

const char *SBEventGetCStringFromEvent(SBEventRef event) {
  return lldb::SBEvent::GetCStringFromEvent(
      *reinterpret_cast<SBEvent *>(event));
}

bool SBEventGetDescription(SBEventRef instance, SBStreamRef description) {
  SBEvent *unwrapped = reinterpret_cast<SBEvent *>(instance);
  return unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description));
}

#ifdef __cplusplus
}
#endif
