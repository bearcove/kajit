//===-- SBBroadcasterBinding.cpp --------------------------------*- C++ -*-===//
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

SBBroadcasterRef CreateSBBroadcaster() {
  return reinterpret_cast<SBBroadcasterRef>(new SBBroadcaster());
}

SBBroadcasterRef CreateSBBroadcaster2(const char *name) {
  return reinterpret_cast<SBBroadcasterRef>(new SBBroadcaster(name));
}

SBBroadcasterRef CloneSBBroadcaster(SBBroadcasterRef instance) {
  return reinterpret_cast<SBBroadcasterRef>(
      new SBBroadcaster(*reinterpret_cast<SBBroadcaster *>(instance)));
}

void DisposeSBBroadcaster(SBBroadcasterRef instance) {
  delete reinterpret_cast<SBBroadcaster *>(instance);
}

bool SBBroadcasterIsValid(SBBroadcasterRef instance) {
  SBBroadcaster *unwrapped = reinterpret_cast<SBBroadcaster *>(instance);
  return unwrapped->IsValid();
}

void SBBroadcasterClear(SBBroadcasterRef instance) {
  SBBroadcaster *unwrapped = reinterpret_cast<SBBroadcaster *>(instance);
  unwrapped->Clear();
}

void SBBroadcasterBroadcastEventByType(SBBroadcasterRef instance,
                                       uint32_t event_type, bool unique) {
  SBBroadcaster *unwrapped = reinterpret_cast<SBBroadcaster *>(instance);
  unwrapped->BroadcastEventByType(event_type, unique);
}

void SBBroadcasterBroadcastEvent(SBBroadcasterRef instance, SBEventRef event,
                                 bool unique) {
  SBBroadcaster *unwrapped = reinterpret_cast<SBBroadcaster *>(instance);
  unwrapped->BroadcastEvent(*reinterpret_cast<SBEvent *>(event), unique);
}

void SBBroadcasterAddInitialEventsToListener(SBBroadcasterRef instance,
                                             SBListenerRef listener,
                                             uint32_t requested_events) {
  SBBroadcaster *unwrapped = reinterpret_cast<SBBroadcaster *>(instance);
  unwrapped->AddInitialEventsToListener(
      *reinterpret_cast<SBListener *>(listener), requested_events);
}

uint32_t SBBroadcasterAddListener(SBBroadcasterRef instance,
                                  SBListenerRef listener, uint32_t event_mask) {
  SBBroadcaster *unwrapped = reinterpret_cast<SBBroadcaster *>(instance);
  return unwrapped->AddListener(*reinterpret_cast<SBListener *>(listener),
                                event_mask);
}

const char *SBBroadcasterGetName(SBBroadcasterRef instance) {
  SBBroadcaster *unwrapped = reinterpret_cast<SBBroadcaster *>(instance);
  return unwrapped->GetName();
}

bool SBBroadcasterEventTypeHasListeners(SBBroadcasterRef instance,
                                        uint32_t event_type) {
  SBBroadcaster *unwrapped = reinterpret_cast<SBBroadcaster *>(instance);
  return unwrapped->EventTypeHasListeners(event_type);
}

bool SBBroadcasterRemoveListener(SBBroadcasterRef instance,
                                 SBListenerRef listener, uint32_t event_mask) {
  SBBroadcaster *unwrapped = reinterpret_cast<SBBroadcaster *>(instance);
  return unwrapped->RemoveListener(*reinterpret_cast<SBListener *>(listener),
                                   event_mask);
}

#ifdef __cplusplus
}
#endif
