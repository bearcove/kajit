//===-- SBEventBinding.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBEventBinding_h_
#define LLDB_SBEventBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBEventRef CreateSBEvent(void);

LLDB_API SBEventRef CreateSBEvent2(uint32_t event, const char *cstr,
                                   uint32_t cstr_len);

LLDB_API SBEventRef CloneSBEvent(SBEventRef instance);

LLDB_API void DisposeSBEvent(SBEventRef instance);

LLDB_API bool SBEventIsValid(SBEventRef instance);

LLDB_API const char *SBEventGetDataFlavor(SBEventRef instance);

LLDB_API uint32_t SBEventGetType(SBEventRef instance);

LLDB_API SBBroadcasterRef SBEventGetBroadcaster(SBEventRef instance);

LLDB_API const char *SBEventGetBroadcasterClass(SBEventRef instance);

LLDB_API bool SBEventBroadcasterMatchesPtr(SBEventRef instance,
                                           SBBroadcasterRef broadcaster);

LLDB_API bool SBEventBroadcasterMatchesRef(SBEventRef instance,
                                           SBBroadcasterRef broadcaster);

LLDB_API void SBEventClear(SBEventRef instance);

LLDB_API const char *SBEventGetCStringFromEvent(SBEventRef event);

LLDB_API bool SBEventGetDescription(SBEventRef instance,
                                    SBStreamRef description);

LLDB_API bool SBEventGetDescription(SBEventRef instance,
                                    SBStreamRef description);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBEventBinding_h_
