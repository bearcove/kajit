//===-- SBBroadcasterBinding.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBBroadcasterBinding_h_
#define LLDB_SBBroadcasterBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBBroadcasterRef CreateSBBroadcaster(void);

LLDB_API SBBroadcasterRef CreateSBBroadcaster2(const char *name);

LLDB_API SBBroadcasterRef CloneSBBroadcaster(SBBroadcasterRef instance);

LLDB_API void DisposeSBBroadcaster(SBBroadcasterRef instance);

LLDB_API bool SBBroadcasterIsValid(SBBroadcasterRef instance);

LLDB_API void SBBroadcasterClear(SBBroadcasterRef instance);

LLDB_API void SBBroadcasterBroadcastEventByType(SBBroadcasterRef instance,
                                                uint32_t event_type,
                                                bool unique);

LLDB_API void SBBroadcasterBroadcastEvent(SBBroadcasterRef instance,
                                          SBEventRef event, bool unique);

LLDB_API void
SBBroadcasterAddInitialEventsToListener(SBBroadcasterRef instance,
                                        SBListenerRef listener,
                                        uint32_t requested_events);

LLDB_API uint32_t SBBroadcasterAddListener(SBBroadcasterRef instance,
                                           SBListenerRef listener,
                                           uint32_t event_mask);

LLDB_API const char *SBBroadcasterGetName(SBBroadcasterRef instance);

LLDB_API bool SBBroadcasterEventTypeHasListeners(SBBroadcasterRef instance,
                                                 uint32_t event_type);

LLDB_API bool SBBroadcasterRemoveListener(SBBroadcasterRef instance,
                                          SBListenerRef listener,
                                          uint32_t event_mask);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBBroadcasterBinding_h_
