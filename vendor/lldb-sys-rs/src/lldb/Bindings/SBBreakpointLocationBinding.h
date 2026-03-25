//===-- SBBreakpointLocationBinding.h ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBBreakpointLocationBinding_h_
#define LLDB_SBBreakpointLocationBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBBreakpointLocationRef CreateSBBreakpointLocation(void);

LLDB_API SBBreakpointLocationRef
CloneSBBreakpointLocation(SBBreakpointLocationRef instance);

LLDB_API void DisposeSBBreakpointLocation(SBBreakpointLocationRef instance);

LLDB_API int SBBreakpointLocationGetID(SBBreakpointLocationRef instance);

LLDB_API bool SBBreakpointLocationIsValid(SBBreakpointLocationRef instance);

LLDB_API SBAddressRef
SBBreakpointLocationGetAddress(SBBreakpointLocationRef instance);

LLDB_API lldb_addr_t
SBBreakpointLocationGetLoadAddress(SBBreakpointLocationRef instance);

LLDB_API void SBBreakpointLocationSetEnabled(SBBreakpointLocationRef instance,
                                             bool enabled);

LLDB_API bool SBBreakpointLocationIsEnabled(SBBreakpointLocationRef instance);

LLDB_API uint32_t
SBBreakpointLocationGetHitCount(SBBreakpointLocationRef instance);

LLDB_API uint32_t
SBBreakpointLocationGetIgnoreCount(SBBreakpointLocationRef instance);

LLDB_API void
SBBreakpointLocationSetIgnoreCount(SBBreakpointLocationRef instance,
                                   uint32_t n);

LLDB_API void SBBreakpointLocationSetCondition(SBBreakpointLocationRef instance,
                                               const char *condition);

LLDB_API const char *
SBBreakpointLocationGetCondition(SBBreakpointLocationRef instance);

LLDB_API void
SBBreakpointLocationSetAutoContinue(SBBreakpointLocationRef instance,
                                    bool auto_continue);

LLDB_API bool
SBBreakpointLocationGetAutoContinue(SBBreakpointLocationRef instance);

LLDB_API void
SBBreakpointLocationSetCallback(SBBreakpointLocationRef instance,
                                lldb::SBBreakpointHitCallback callback,
                                void *baton);

LLDB_API SBErrorRef SBBreakpointLocationSetScriptCallbackFunction(
    SBBreakpointLocationRef instance, const char *callback_function_name,
    SBStructuredDataRef extra_args);

LLDB_API SBErrorRef SBBreakpointLocationSetScriptCallbackBody(
    SBBreakpointLocationRef instance, const char *script_body_text);

LLDB_API void
SBBreakpointLocationSetCommandLineCommands(SBBreakpointLocationRef instance,
                                           SBStringListRef commands);

LLDB_API bool
SBBreakpointLocationGetCommandLineCommands(SBBreakpointLocationRef instance,
                                           SBStringListRef commands);

LLDB_API void SBBreakpointLocationSetThreadID(SBBreakpointLocationRef instance,
                                              lldb_tid_t sb_thread_id);

LLDB_API lldb_tid_t
SBBreakpointLocationGetThreadID(SBBreakpointLocationRef instance);

LLDB_API void
SBBreakpointLocationSetThreadIndex(SBBreakpointLocationRef instance,
                                   uint32_t index);

LLDB_API uint32_t
SBBreakpointLocationGetThreadIndex(SBBreakpointLocationRef instance);

LLDB_API void
SBBreakpointLocationSetThreadName(SBBreakpointLocationRef instance,
                                  const char *thread_name);

LLDB_API const char *
SBBreakpointLocationGetThreadName(SBBreakpointLocationRef instance);

LLDB_API void SBBreakpointLocationSetQueueName(SBBreakpointLocationRef instance,
                                               const char *queue_name);

LLDB_API const char *
SBBreakpointLocationGetQueueName(SBBreakpointLocationRef instance);

LLDB_API bool SBBreakpointLocationIsResolved(SBBreakpointLocationRef instance);

LLDB_API bool
SBBreakpointLocationGetDescription(SBBreakpointLocationRef instance,
                                   SBStreamRef description,
                                   ENUM(DescriptionLevel) level);

LLDB_API SBBreakpointRef
SBBreakpointLocationGetBreakpoint(SBBreakpointLocationRef instance);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBBreakpointLocationBinding_h_
