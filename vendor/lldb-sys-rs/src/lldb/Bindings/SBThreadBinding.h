//===-- SBThreadBinding.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBThreadBinding_h_
#define LLDB_SBThreadBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API const char *SBThreadGetBroadcasterClassName(void);

LLDB_API SBThreadRef CreateSBThread(void);

LLDB_API SBThreadRef CloneSBThread(SBThreadRef instance);

LLDB_API void DisposeSBThread(SBThreadRef instance);

LLDB_API SBQueueRef SBThreadGetQueue(SBThreadRef instance);

LLDB_API bool SBThreadIsValid(SBThreadRef instance);

LLDB_API void SBThreadClear(SBThreadRef instance);

LLDB_API ENUM(StopReason) SBThreadGetStopReason(SBThreadRef instance);

LLDB_API size_t SBThreadGetStopReasonDataCount(SBThreadRef instance);

LLDB_API uint64_t SBThreadGetStopReasonDataAtIndex(SBThreadRef instance,
                                                   uint32_t idx);

LLDB_API bool SBThreadGetStopReasonExtendedInfoAsJSON(SBThreadRef instance,
                                                      SBStreamRef stream);

LLDB_API SBThreadCollectionRef SBThreadGetStopReasonExtendedBacktaces(
    SBThreadRef instance, ENUM(InstrumentationRuntimeType) type);

LLDB_API size_t SBThreadGetStopDescription(SBThreadRef instance, char *dst,
                                           size_t dst_len);

LLDB_API SBValueRef SBThreadGetStopReturnValue(SBThreadRef instance);

LLDB_API lldb_tid_t SBThreadGetThreadID(SBThreadRef instance);

LLDB_API uint32_t SBThreadGetIndexID(SBThreadRef instance);

LLDB_API const char *SBThreadGetName(SBThreadRef instance);

LLDB_API const char *SBThreadGetQueueName(SBThreadRef instance);

LLDB_API lldb_queue_id_t SBThreadGetQueueID(SBThreadRef instance);

LLDB_API bool SBThreadGetInfoItemByPathAsString(SBThreadRef instance,
                                                const char *path,
                                                SBStreamRef strm);

LLDB_API void SBThreadStepOver(SBThreadRef instance,
                               ENUM(RunMode) stop_other_threads,
                               SBErrorRef error);

LLDB_API void SBThreadStepInto(SBThreadRef instance,
                               ENUM(RunMode) stop_other_threads);

LLDB_API void SBThreadStepInto2(SBThreadRef instance, const char *target_name,
                                ENUM(RunMode) stop_other_threads);

LLDB_API void SBThreadStepInto3(SBThreadRef instance, const char *target_name,
                                uint32_t end_line, SBErrorRef error,
                                ENUM(RunMode) stop_other_threads);

LLDB_API void SBThreadStepOut(SBThreadRef instance, SBErrorRef error);

LLDB_API void SBThreadStepOutOfFrame(SBThreadRef instance, SBFrameRef frame,
                                     SBErrorRef error);

LLDB_API void SBThreadStepInstruction(SBThreadRef instance, bool step_over,
                                      SBErrorRef error);

LLDB_API SBErrorRef SBThreadStepOverUntil(SBThreadRef instance,
                                          SBFrameRef frame,
                                          SBFileSpecRef file_spec,
                                          uint32_t line);

LLDB_API SBErrorRef SBThreadStepUsingScriptedThreadPlan(
    SBThreadRef instance, const char *script_class_name,
    SBStructuredDataRef args_data, bool resume_immediately);

LLDB_API SBErrorRef SBThreadJumpToLine(SBThreadRef instance,
                                       SBFileSpecRef file_spec, uint32_t line);

LLDB_API void SBThreadRunToAddress(SBThreadRef instance, lldb_addr_t addr,
                                   SBErrorRef error);

LLDB_API SBErrorRef SBThreadReturnFromFrame(SBThreadRef instance,
                                            SBFrameRef frame,
                                            SBValueRef return_value);

LLDB_API SBErrorRef SBThreadUnwindInnermostExpression(SBThreadRef instance);

LLDB_API bool SBThreadSuspend(SBThreadRef instance, SBErrorRef error);

LLDB_API bool SBThreadResume(SBThreadRef instance, SBErrorRef error);

LLDB_API bool SBThreadIsSuspended(SBThreadRef instance);

LLDB_API bool SBThreadIsStopped(SBThreadRef instance);

LLDB_API uint32_t SBThreadGetNumFrames(SBThreadRef instance);

LLDB_API SBFrameRef SBThreadGetFrameAtIndex(SBThreadRef instance, uint32_t idx);

LLDB_API SBFrameRef SBThreadGetSelectedFrame(SBThreadRef instance);

LLDB_API SBFrameRef SBThreadSetSelectedFrame(SBThreadRef instance,
                                             uint32_t frame_idx);

LLDB_API bool SBThreadEventIsThreadEvent(SBEventRef event);

LLDB_API SBFrameRef SBThreadGetStackFrameFromEvent(SBEventRef event);

LLDB_API SBThreadRef SBThreadGetThreadFromEvent(SBEventRef event);

LLDB_API SBProcessRef SBThreadGetProcess(SBThreadRef instance);

LLDB_API bool SBThreadGetDescription(SBThreadRef instance,
                                     SBStreamRef description);

LLDB_API bool SBThreadGetStatus(SBThreadRef instance, SBStreamRef status);

LLDB_API SBThreadRef SBThreadGetExtendedBacktraceThread(SBThreadRef instance,
                                                        const char *type);

LLDB_API uint32_t
SBThreadGetExtendedBacktraceOriginatingIndexID(SBThreadRef instance);

LLDB_API SBValueRef SBThreadGetCurrentException(SBThreadRef instance);

LLDB_API SBThreadRef SBThreadGetCurrentExceptionBacktrace(SBThreadRef instance);

LLDB_API bool SBThreadSafeToCallFunctions(SBThreadRef instance);

LLDB_API SBValueRef SBThreadGetSiginfo(SBThreadRef instance);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBThreadBinding_h_
