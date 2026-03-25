//===-- SBListenerBinding.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBListenerBinding_h_
#define LLDB_SBListenerBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBListenerRef CreateSBListener(void);

LLDB_API SBListenerRef CreateSBListener2(const char *name);

LLDB_API SBListenerRef CloneSBListener(SBListenerRef instance);

LLDB_API void DisposeSBListener(SBListenerRef instance);

LLDB_API void SBListenerAddEvent(SBListenerRef instance, SBEventRef event);

LLDB_API void SBListenerClear(SBListenerRef instance);

LLDB_API bool SBListenerIsValid(SBListenerRef instance);

LLDB_API uint32_t SBListenerStartListeningForEventClass(
    SBListenerRef instance, SBDebuggerRef debugger,
    const char *broadcaster_class, uint32_t event_mask);

LLDB_API bool SBListenerStopListeningForEventClass(
    SBListenerRef instance, SBDebuggerRef debugger,
    const char *broadcaster_class, uint32_t event_mask);

LLDB_API uint32_t SBListenerStartListeningForEvents(
    SBListenerRef instance, SBBroadcasterRef broadcaster, uint32_t event_mask);

LLDB_API bool SBListenerStopListeningForEvents(SBListenerRef instance,
                                               SBBroadcasterRef broadcaster,
                                               uint32_t event_mask);

LLDB_API bool SBListenerWaitForEvent(SBListenerRef instance,
                                     uint32_t num_seconds, SBEventRef event);

LLDB_API bool SBListenerWaitForEventForBroadcaster(SBListenerRef instance,
                                                   uint32_t num_seconds,
                                                   SBBroadcasterRef broadcaster,
                                                   SBEventRef sb_event);

LLDB_API bool SBListenerWaitForEventForBroadcasterWithType(
    SBListenerRef instance, uint32_t num_seconds, SBBroadcasterRef broadcaster,
    uint32_t event_type_mask, SBEventRef sb_event);

LLDB_API bool SBListenerPeekAtNextEvent(SBListenerRef instance,
                                        SBEventRef sb_event);

LLDB_API bool SBListenerPeekAtNextEventForBroadcaster(
    SBListenerRef instance, SBBroadcasterRef broadcaster, SBEventRef sb_event);

LLDB_API bool SBListenerPeekAtNextEventForBroadcasterWithType(
    SBListenerRef instance, SBBroadcasterRef broadcaster,
    uint32_t event_type_mask, SBEventRef sb_event);

LLDB_API bool SBListenerGetNextEvent(SBListenerRef instance,
                                     SBEventRef sb_event);

LLDB_API bool SBListenerGetNextEventForBroadcaster(SBListenerRef instance,
                                                   SBBroadcasterRef broadcaster,
                                                   SBEventRef sb_event);

LLDB_API bool SBListenerGetNextEventForBroadcasterWithType(
    SBListenerRef instance, SBBroadcasterRef broadcaster,
    uint32_t event_type_mask, SBEventRef sb_event);

LLDB_API bool SBListenerHandleBroadcastEvent(SBListenerRef instance,
                                             SBEventRef event);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBListenerBinding_h_
