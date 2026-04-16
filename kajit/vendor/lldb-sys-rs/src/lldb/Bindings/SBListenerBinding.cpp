//===-- SBListenerBinding.cpp -----------------------------------*- C++ -*-===//
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

SBListenerRef CreateSBListener() {
  return reinterpret_cast<SBListenerRef>(new SBListener());
}

SBListenerRef CreateSBListener2(const char *name) {
  return reinterpret_cast<SBListenerRef>(new SBListener(name));
}

SBListenerRef CloneSBListener(SBListenerRef instance) {
  return reinterpret_cast<SBListenerRef>(
      new SBListener(*reinterpret_cast<SBListener *>(instance)));
}

void DisposeSBListener(SBListenerRef instance) {
  delete reinterpret_cast<SBListener *>(instance);
}

void SBListenerAddEvent(SBListenerRef instance, SBEventRef event) {
  SBListener *unwrapped = reinterpret_cast<SBListener *>(instance);
  unwrapped->AddEvent(*reinterpret_cast<SBEvent *>(event));
}

void SBListenerClear(SBListenerRef instance) {
  SBListener *unwrapped = reinterpret_cast<SBListener *>(instance);
  unwrapped->Clear();
}

bool SBListenerIsValid(SBListenerRef instance) {
  SBListener *unwrapped = reinterpret_cast<SBListener *>(instance);
  return unwrapped->IsValid();
}

uint32_t SBListenerStartListeningForEventClass(SBListenerRef instance,
                                               SBDebuggerRef debugger,
                                               const char *broadcaster_class,
                                               uint32_t event_mask) {
  SBListener *unwrapped = reinterpret_cast<SBListener *>(instance);
  return unwrapped->StartListeningForEventClass(
      *reinterpret_cast<SBDebugger *>(debugger), broadcaster_class, event_mask);
}

bool SBListenerStopListeningForEventClass(SBListenerRef instance,
                                          SBDebuggerRef debugger,
                                          const char *broadcaster_class,
                                          uint32_t event_mask) {
  SBListener *unwrapped = reinterpret_cast<SBListener *>(instance);
  return unwrapped->StopListeningForEventClass(
      *reinterpret_cast<SBDebugger *>(debugger), broadcaster_class, event_mask);
}

uint32_t SBListenerStartListeningForEvents(SBListenerRef instance,
                                           SBBroadcasterRef broadcaster,
                                           uint32_t event_mask) {
  SBListener *unwrapped = reinterpret_cast<SBListener *>(instance);
  return unwrapped->StartListeningForEvents(
      *reinterpret_cast<SBBroadcaster *>(broadcaster), event_mask);
}

bool SBListenerStopListeningForEvents(SBListenerRef instance,
                                      SBBroadcasterRef broadcaster,
                                      uint32_t event_mask) {
  SBListener *unwrapped = reinterpret_cast<SBListener *>(instance);
  return unwrapped->StopListeningForEvents(
      *reinterpret_cast<SBBroadcaster *>(broadcaster), event_mask);
}

bool SBListenerWaitForEvent(SBListenerRef instance, uint32_t num_seconds,
                            SBEventRef event) {
  SBListener *unwrapped = reinterpret_cast<SBListener *>(instance);
  return unwrapped->WaitForEvent(num_seconds,
                                 *reinterpret_cast<SBEvent *>(event));
}

bool SBListenerWaitForEventForBroadcaster(SBListenerRef instance,
                                          uint32_t num_seconds,
                                          SBBroadcasterRef broadcaster,
                                          SBEventRef sb_event) {
  SBListener *unwrapped = reinterpret_cast<SBListener *>(instance);
  return unwrapped->WaitForEventForBroadcaster(
      num_seconds, *reinterpret_cast<SBBroadcaster *>(broadcaster),
      *reinterpret_cast<SBEvent *>(sb_event));
}

bool SBListenerWaitForEventForBroadcasterWithType(SBListenerRef instance,
                                                  uint32_t num_seconds,
                                                  SBBroadcasterRef broadcaster,
                                                  uint32_t event_type_mask,
                                                  SBEventRef sb_event) {
  SBListener *unwrapped = reinterpret_cast<SBListener *>(instance);
  return unwrapped->WaitForEventForBroadcasterWithType(
      num_seconds, *reinterpret_cast<SBBroadcaster *>(broadcaster),
      event_type_mask, *reinterpret_cast<SBEvent *>(sb_event));
}

bool SBListenerPeekAtNextEvent(SBListenerRef instance, SBEventRef sb_event) {
  SBListener *unwrapped = reinterpret_cast<SBListener *>(instance);
  return unwrapped->PeekAtNextEvent(*reinterpret_cast<SBEvent *>(sb_event));
}

bool SBListenerPeekAtNextEventForBroadcaster(SBListenerRef instance,
                                             SBBroadcasterRef broadcaster,
                                             SBEventRef sb_event) {
  SBListener *unwrapped = reinterpret_cast<SBListener *>(instance);
  return unwrapped->PeekAtNextEventForBroadcaster(
      *reinterpret_cast<SBBroadcaster *>(broadcaster),
      *reinterpret_cast<SBEvent *>(sb_event));
}

bool SBListenerPeekAtNextEventForBroadcasterWithType(
    SBListenerRef instance, SBBroadcasterRef broadcaster,
    uint32_t event_type_mask, SBEventRef sb_event) {
  SBListener *unwrapped = reinterpret_cast<SBListener *>(instance);
  return unwrapped->PeekAtNextEventForBroadcasterWithType(
      *reinterpret_cast<SBBroadcaster *>(broadcaster), event_type_mask,
      *reinterpret_cast<SBEvent *>(sb_event));
}

bool SBListenerGetNextEvent(SBListenerRef instance, SBEventRef sb_event) {
  SBListener *unwrapped = reinterpret_cast<SBListener *>(instance);
  return unwrapped->GetNextEvent(*reinterpret_cast<SBEvent *>(sb_event));
}

bool SBListenerGetNextEventForBroadcaster(SBListenerRef instance,
                                          SBBroadcasterRef broadcaster,
                                          SBEventRef sb_event) {
  SBListener *unwrapped = reinterpret_cast<SBListener *>(instance);
  return unwrapped->GetNextEventForBroadcaster(
      *reinterpret_cast<SBBroadcaster *>(broadcaster),
      *reinterpret_cast<SBEvent *>(sb_event));
}

bool SBListenerGetNextEventForBroadcasterWithType(SBListenerRef instance,
                                                  SBBroadcasterRef broadcaster,
                                                  uint32_t event_type_mask,
                                                  SBEventRef sb_event) {
  SBListener *unwrapped = reinterpret_cast<SBListener *>(instance);
  return unwrapped->GetNextEventForBroadcasterWithType(
      *reinterpret_cast<SBBroadcaster *>(broadcaster), event_type_mask,
      *reinterpret_cast<SBEvent *>(sb_event));
}

bool SBListenerHandleBroadcastEvent(SBListenerRef instance, SBEventRef event) {
  SBListener *unwrapped = reinterpret_cast<SBListener *>(instance);
  return unwrapped->HandleBroadcastEvent(*reinterpret_cast<SBEvent *>(event));
}

#ifdef __cplusplus
}
#endif
