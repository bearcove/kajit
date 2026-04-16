//===-- SBThreadBinding.cpp -------------------------------------*- C++ -*-===//
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

const char *SBThreadGetBroadcasterClassName() {
  return lldb::SBThread::GetBroadcasterClassName();
}

SBThreadRef CreateSBThread() {
  return reinterpret_cast<SBThreadRef>(new SBThread());
}

SBThreadRef CreateSBThread2(SBThreadRef thread) {
  return reinterpret_cast<SBThreadRef>(
      new SBThread(*reinterpret_cast<SBThread *>(thread)));
}

SBThreadRef CloneSBThread(SBThreadRef instance) {
  return reinterpret_cast<SBThreadRef>(
      new SBThread(*reinterpret_cast<SBThread *>(instance)));
}

void DisposeSBThread(SBThreadRef instance) {
  delete reinterpret_cast<SBThread *>(instance);
}

SBQueueRef SBThreadGetQueue(SBThreadRef instance) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return reinterpret_cast<SBQueueRef>(new SBQueue(unwrapped->GetQueue()));
}

bool SBThreadIsValid(SBThreadRef instance) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return unwrapped->IsValid();
}

void SBThreadClear(SBThreadRef instance) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  unwrapped->Clear();
}

enum lldb::StopReason SBThreadGetStopReason(SBThreadRef instance) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return unwrapped->GetStopReason();
}

size_t SBThreadGetStopReasonDataCount(SBThreadRef instance) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return unwrapped->GetStopReasonDataCount();
}

uint64_t SBThreadGetStopReasonDataAtIndex(SBThreadRef instance, uint32_t idx) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return unwrapped->GetStopReasonDataAtIndex(idx);
}

bool SBThreadGetStopReasonExtendedInfoAsJSON(SBThreadRef instance,
                                             SBStreamRef stream) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return unwrapped->GetStopReasonExtendedInfoAsJSON(
      *reinterpret_cast<SBStream *>(stream));
}

SBThreadCollectionRef
SBThreadGetStopReasonExtendedBacktaces(SBThreadRef instance,
                                       ENUM(InstrumentationRuntimeType) type) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return reinterpret_cast<SBThreadCollectionRef>(
      new SBThreadCollection(unwrapped->GetStopReasonExtendedBacktraces(type)));
}

size_t SBThreadGetStopDescription(SBThreadRef instance, char *dst,
                                  size_t dst_len) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return unwrapped->GetStopDescription(dst, dst_len);
}

SBValueRef SBThreadGetStopReturnValue(SBThreadRef instance) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return reinterpret_cast<SBValueRef>(
      new SBValue(unwrapped->GetStopReturnValue()));
}

lldb_tid_t SBThreadGetThreadID(SBThreadRef instance) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return unwrapped->GetThreadID();
}

uint32_t SBThreadGetIndexID(SBThreadRef instance) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return unwrapped->GetIndexID();
}

const char *SBThreadGetName(SBThreadRef instance) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return unwrapped->GetName();
}

const char *SBThreadGetQueueName(SBThreadRef instance) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return unwrapped->GetQueueName();
}

lldb_queue_id_t SBThreadGetQueueID(SBThreadRef instance) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return unwrapped->GetQueueID();
}

bool SBThreadGetInfoItemByPathAsString(SBThreadRef instance, const char *path,
                                       SBStreamRef strm) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return unwrapped->GetInfoItemByPathAsString(
      path, *reinterpret_cast<SBStream *>(strm));
}

void SBThreadStepOver(SBThreadRef instance, lldb::RunMode stop_other_threads,
                      SBErrorRef error) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  unwrapped->StepOver(stop_other_threads, *reinterpret_cast<SBError *>(error));
}

void SBThreadStepInto(SBThreadRef instance, lldb::RunMode stop_other_threads) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  unwrapped->StepInto(stop_other_threads);
}

void SBThreadStepInto2(SBThreadRef instance, const char *target_name,
                       lldb::RunMode stop_other_threads) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  unwrapped->StepInto(target_name, stop_other_threads);
}

void SBThreadStepInto3(SBThreadRef instance, const char *target_name,
                       uint32_t end_line, SBErrorRef error,
                       ENUM(RunMode) stop_other_threads) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  unwrapped->StepInto(target_name, end_line,
                      *reinterpret_cast<SBError *>(error), stop_other_threads);
}

void SBThreadStepOut(SBThreadRef instance, SBErrorRef error) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  unwrapped->StepOut(*reinterpret_cast<SBError *>(error));
}

void SBThreadStepOutOfFrame(SBThreadRef instance, SBFrameRef frame,
                            SBErrorRef error) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  unwrapped->StepOutOfFrame(*reinterpret_cast<SBFrame *>(frame),
                            *reinterpret_cast<SBError *>(error));
}

void SBThreadStepInstruction(SBThreadRef instance, bool step_over,
                             SBErrorRef error) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  unwrapped->StepInstruction(step_over, *reinterpret_cast<SBError *>(error));
}

SBErrorRef SBThreadStepOverUntil(SBThreadRef instance, SBFrameRef frame,
                                 SBFileSpecRef file_spec, uint32_t line) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return reinterpret_cast<SBErrorRef>(new SBError(unwrapped->StepOverUntil(
      *reinterpret_cast<SBFrame *>(frame),
      *reinterpret_cast<SBFileSpec *>(file_spec), line)));
}

SBErrorRef SBThreadStepUsingScriptedThreadPlan(SBThreadRef instance,
                                               const char *script_class_name,
                                               SBStructuredDataRef args_data,
                                               bool resume_immediately) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return reinterpret_cast<SBErrorRef>(
      new SBError(unwrapped->StepUsingScriptedThreadPlan(
          script_class_name, *reinterpret_cast<SBStructuredData *>(args_data),
          resume_immediately)));
}

SBErrorRef SBThreadJumpToLine(SBThreadRef instance, SBFileSpecRef file_spec,
                              uint32_t line) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return reinterpret_cast<SBErrorRef>(new SBError(
      unwrapped->JumpToLine(*reinterpret_cast<SBFileSpec *>(file_spec), line)));
}

void SBThreadRunToAddress(SBThreadRef instance, lldb_addr_t addr,
                          SBErrorRef error) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  unwrapped->RunToAddress(addr, *reinterpret_cast<SBError *>(error));
}

SBErrorRef SBThreadReturnFromFrame(SBThreadRef instance, SBFrameRef frame,
                                   SBValueRef return_value) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return reinterpret_cast<SBErrorRef>(new SBError(
      unwrapped->ReturnFromFrame(*reinterpret_cast<SBFrame *>(frame),
                                 *reinterpret_cast<SBValue *>(return_value))));
}

SBErrorRef SBThreadUnwindInnermostExpression(SBThreadRef instance) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return reinterpret_cast<SBErrorRef>(
      new SBError(unwrapped->UnwindInnermostExpression()));
}

bool SBThreadSuspend(SBThreadRef instance, SBErrorRef error) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return unwrapped->Suspend(*reinterpret_cast<SBError *>(error));
}

bool SBThreadResume(SBThreadRef instance, SBErrorRef error) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return unwrapped->Resume(*reinterpret_cast<SBError *>(error));
}

bool SBThreadIsSuspended(SBThreadRef instance) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return unwrapped->IsSuspended();
}

bool SBThreadIsStopped(SBThreadRef instance) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return unwrapped->IsStopped();
}

uint32_t SBThreadGetNumFrames(SBThreadRef instance) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return unwrapped->GetNumFrames();
}

SBFrameRef SBThreadGetFrameAtIndex(SBThreadRef instance, uint32_t idx) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return reinterpret_cast<SBFrameRef>(
      new SBFrame(unwrapped->GetFrameAtIndex(idx)));
}

SBFrameRef SBThreadGetSelectedFrame(SBThreadRef instance) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return reinterpret_cast<SBFrameRef>(
      new SBFrame(unwrapped->GetSelectedFrame()));
}

SBFrameRef SBThreadSetSelectedFrame(SBThreadRef instance, uint32_t frame_idx) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return reinterpret_cast<SBFrameRef>(
      new SBFrame(unwrapped->SetSelectedFrame(frame_idx)));
}

bool SBThreadEventIsThreadEvent(SBEventRef event) {
  return lldb::SBThread::EventIsThreadEvent(
      *reinterpret_cast<SBEvent *>(event));
}

SBFrameRef SBThreadGetStackFrameFromEvent(SBEventRef event) {
  return reinterpret_cast<SBFrameRef>(
      new SBFrame(lldb::SBThread::GetStackFrameFromEvent(
          *reinterpret_cast<SBEvent *>(event))));
}

SBThreadRef SBThreadGetThreadFromEvent(SBEventRef event) {
  return reinterpret_cast<SBThreadRef>(new SBThread(
      lldb::SBThread::GetThreadFromEvent(*reinterpret_cast<SBEvent *>(event))));
}

SBProcessRef SBThreadGetProcess(SBThreadRef instance) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return reinterpret_cast<SBProcessRef>(new SBProcess(unwrapped->GetProcess()));
}

bool SBThreadGetDescription(SBThreadRef instance, SBStreamRef description) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description));
}

bool SBThreadGetStatus(SBThreadRef instance, SBStreamRef status) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return unwrapped->GetStatus(*reinterpret_cast<SBStream *>(status));
}

SBThreadRef SBThreadGetExtendedBacktraceThread(SBThreadRef instance,
                                               const char *type) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return reinterpret_cast<SBThreadRef>(
      new SBThread(unwrapped->GetExtendedBacktraceThread(type)));
}

uint32_t SBThreadGetExtendedBacktraceOriginatingIndexID(SBThreadRef instance) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return unwrapped->GetExtendedBacktraceOriginatingIndexID();
}

SBValueRef SBThreadGetCurrentException(SBThreadRef instance) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return reinterpret_cast<SBValueRef>(
      new SBValue(unwrapped->GetCurrentException()));
}

SBThreadRef SBThreadGetCurrentExceptionBacktrace(SBThreadRef instance) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return reinterpret_cast<SBThreadRef>(
      new SBThread(unwrapped->GetCurrentExceptionBacktrace()));
}

bool SBThreadSafeToCallFunctions(SBThreadRef instance) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return unwrapped->SafeToCallFunctions();
}

SBValueRef SBThreadGetSiginfo(SBThreadRef instance) {
  SBThread *unwrapped = reinterpret_cast<SBThread *>(instance);
  return reinterpret_cast<SBValueRef>(new SBValue(unwrapped->GetSiginfo()));
}

#ifdef __cplusplus
}
#endif
