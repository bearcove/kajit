//===-- SBBreakpointNameBinding.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBBreakpointNameBinding_h_
#define LLDB_SBBreakpointNameBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBBreakpointNameRef CreateSBBreakpointName(void);

LLDB_API SBBreakpointNameRef
CreateSBBreakpointNameFromTarget(SBTargetRef target, const char *name);

LLDB_API SBBreakpointNameRef CreateSBBreakpointNameFromBreakpoint(
    SBBreakpointRef breakpoint, const char *name);

LLDB_API SBBreakpointNameRef
CloneSBBreakpointName(SBBreakpointNameRef instance);

LLDB_API void DisposeSBBreakpointName(SBBreakpointNameRef instance);

LLDB_API bool SBBreakpointNameIsValid(SBBreakpointNameRef instance);

LLDB_API const char *
SBBreakpointNameGetName(SBBreakpointNameRef instance) const;

LLDB_API void SBBreakpointNameSetEnabled(SBBreakpointNameRef instance,
                                         bool enable);

LLDB_API bool SBBreakpointNameIsEnabled(SBBreakpointNameRef instance);

LLDB_API void SBBreakpointNameSetOneShot(SBBreakpointNameRef instance,
                                         bool one_shot);

LLDB_API bool SBBreakpointNameIsOneShot(SBBreakpointNameRef instance) const;

LLDB_API void SBBreakpointNameSetIgnoreCount(SBBreakpointNameRef instance,
                                             uint32_t count);

LLDB_API uint32_t
SBBreakpointNameGetIgnoreCount(SBBreakpointNameRef instance) const;

LLDB_API void SBBreakpointNameSetCondition(SBBreakpointNameRef instance,
                                           const char *condition);

LLDB_API const char *SBBreakpointNameGetCondition(SBBreakpointNameRef instance);

LLDB_API void SBBreakpointNameSetAutoContinue(SBBreakpointNameRef instance,
                                              bool auto_continue);

LLDB_API bool SBBreakpointNameGetAutoContinue(SBBreakpointNameRef instance);

LLDB_API void SBBreakpointNameSetThreadID(SBBreakpointNameRef instance,
                                          lldb_tid_t sb_thread_id);

LLDB_API lldb_tid_t SBBreakpointNameGetThreadID(SBBreakpointNameRef instance);

LLDB_API void SBBreakpointNameSetThreadIndex(SBBreakpointNameRef instance,
                                             uint32_t index);

LLDB_API uint32_t
SBBreakpointNameGetThreadIndex(SBBreakpointNameRef instance) const;

LLDB_API void SBBreakpointNameSetThreadName(SBBreakpointNameRef instance,
                                            const char *thread_name);

LLDB_API const char *
SBBreakpointNameGetThreadName(SBBreakpointNameRef instance) const;

LLDB_API void SBBreakpointNameSetQueueName(SBBreakpointNameRef instance,
                                           const char *queue_name);

LLDB_API const char *
SBBreakpointNameGetQueueName(SBBreakpointNameRef instance) const;

LLDB_API void
SBBreakpointNameSetCallback(SBBreakpointNameRef instance,
                            lldb::SBBreakpointHitCallback callback,
                            void *baton);

LLDB_API SBErrorRef SBBreakpointNameSetScriptCallbackFunction(
    SBBreakpointRef instance, const char *callback_function_name,
    SBStructuredDataRef extra_args);

LLDB_API void
SBBreakpointNameSetCommandLineCommands(SBBreakpointNameRef instance,
                                       SBStringListRef commands);

LLDB_API bool
SBBreakpointNameGetCommandLineCommands(SBBreakpointNameRef instance,
                                       SBStringListRef commands);

LLDB_API SBErrorRef SBBreakpointNameSetScriptCallbackBody(
    SBBreakpointNameRef instance, const char *script_body_text);

LLDB_API const char *
SBBreakpointNameGetHelpString(SBBreakpointNameRef instance) const;

LLDB_API void SBBreakpointNameSetHelpString(SBBreakpointNameRef instance,
                                            const char *help_string);

LLDB_API bool SBBreakpointNameGetAllowList(SBBreakpointNameRef instance) const;

LLDB_API void SBBreakpointNameSetAllowList(SBBreakpointNameRef instance,
                                           bool value);

LLDB_API bool SBBreakpointNameGetAllowDelete(SBBreakpointNameRef instance);

LLDB_API void SBBreakpointNameSetAllowDelete(SBBreakpointNameRef instance,
                                             bool value);

LLDB_API bool SBBreakpointNameGetAllowDisable(SBBreakpointNameRef instance);

LLDB_API void SBBreakpointNameSetAllowDisable(SBBreakpointNameRef instance,
                                              bool value);

LLDB_API bool SBBreakpointNameGetDescription(SBBreakpointNameRef instance,
                                             SBStreamRef description);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBBreakpointNameBinding_h_
