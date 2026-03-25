//===-- SBBreakpointNameBinding.cpp -----------------------------*- C++ -*-===//
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

SBBreakpointNameRef CreateSBBreakpointName() {
  return reinterpret_cast<SBBreakpointNameRef>(new SBBreakpointName());
}

SBBreakpointNameRef CreateSBBreakpointNameFromTarget(SBTargetRef target,
                                                     const char *name) {
  SBTarget *unwrapped_target = reinterpret_cast<SBTarget *>(target);
  return reinterpret_cast<SBBreakpointNameRef>(
      new SBBreakpointName(*unwrapped_target, name));
}

SBBreakpointNameRef
CreateSBBreakpointNameFromBreakpoint(SBBreakpointRef breakpoint,
                                     const char *name) {
  SBBreakpoint *unwrapped_breakpoint =
      reinterpret_cast<SBBreakpoint *>(breakpoint);
  return reinterpret_cast<SBBreakpointNameRef>(
      new SBBreakpointName(*unwrapped_breakpoint, name));
}

SBBreakpointNameRef CloneSBBreakpointName(SBBreakpointNameRef instance) {
  return reinterpret_cast<SBBreakpointNameRef>(
      new SBBreakpointName(*reinterpret_cast<SBBreakpointName *>(instance)));
}

void DisposeSBBreakpointName(SBBreakpointNameRef instance) {
  delete reinterpret_cast<SBBreakpointName *>(instance);
}

bool SBBreakpointNameIsValid(SBBreakpointNameRef instance) {
  SBBreakpointName *unwrapped = reinterpret_cast<SBBreakpointName *>(instance);
  return unwrapped->IsValid();
}

const char *SBBreakpointNameGetName(SBBreakpointNameRef instance) {
  SBBreakpointName *unwrapped = reinterpret_cast<SBBreakpointName *>(instance);
  return unwrapped->GetName();
}

void SBBreakpointNameSetEnabled(SBBreakpointNameRef instance, bool enable) {
  SBBreakpointName *unwrapped = reinterpret_cast<SBBreakpointName *>(instance);
  unwrapped->SetEnabled(enable);
}

bool SBBreakpointNameIsEnabled(SBBreakpointNameRef instance) {
  SBBreakpointName *unwrapped = reinterpret_cast<SBBreakpointName *>(instance);
  return unwrapped->IsEnabled();
}

void SBBreakpointNameSetOneShot(SBBreakpointNameRef instance, bool one_shot) {
  SBBreakpointName *unwrapped = reinterpret_cast<SBBreakpointName *>(instance);
  unwrapped->SetOneShot(one_shot);
}

bool SBBreakpointNameIsOneShot(SBBreakpointNameRef instance) {
  SBBreakpointName *unwrapped = reinterpret_cast<SBBreakpointName *>(instance);
  return unwrapped->IsOneShot();
}

void SBBreakpointNameSetIgnoreCount(SBBreakpointNameRef instance,
                                    uint32_t count) {
  SBBreakpointName *unwrapped = reinterpret_cast<SBBreakpointName *>(instance);
  unwrapped->SetIgnoreCount(count);
}

uint32_t SBBreakpointNameGetIgnoreCount(SBBreakpointNameRef instance) {
  SBBreakpointName *unwrapped = reinterpret_cast<SBBreakpointName *>(instance);
  return unwrapped->GetIgnoreCount();
}

void SBBreakpointNameSetCondition(SBBreakpointNameRef instance,
                                  const char *condition) {
  SBBreakpointName *unwrapped = reinterpret_cast<SBBreakpointName *>(instance);
  unwrapped->SetCondition(condition);
}

const char *SBBreakpointNameGetCondition(SBBreakpointNameRef instance) {
  SBBreakpointName *unwrapped = reinterpret_cast<SBBreakpointName *>(instance);
  return unwrapped->GetCondition();
}

void SBBreakpointNameSetThreadID(SBBreakpointNameRef instance,
                                 lldb_tid_t sb_thread_id) {
  SBBreakpointName *unwrapped = reinterpret_cast<SBBreakpointName *>(instance);
  unwrapped->SetThreadID(sb_thread_id);
}

lldb_tid_t SBBreakpointNameGetThreadID(SBBreakpointNameRef instance) {
  SBBreakpointName *unwrapped = reinterpret_cast<SBBreakpointName *>(instance);
  return unwrapped->GetThreadID();
}

void SBBreakpointNameSetThreadIndex(SBBreakpointNameRef instance,
                                    uint32_t index) {
  SBBreakpointName *unwrapped = reinterpret_cast<SBBreakpointName *>(instance);
  unwrapped->SetThreadIndex(index);
}

uint32_t SBBreakpointNameGetThreadIndex(SBBreakpointNameRef instance) {
  SBBreakpointName *unwrapped = reinterpret_cast<SBBreakpointName *>(instance);
  return unwrapped->GetThreadIndex();
}

void SBBreakpointNameSetThreadName(SBBreakpointNameRef instance,
                                   const char *thread_name) {
  SBBreakpointName *unwrapped = reinterpret_cast<SBBreakpointName *>(instance);
  unwrapped->SetThreadName(thread_name);
}

const char *SBBreakpointNameGetThreadName(SBBreakpointNameRef instance) {
  SBBreakpointName *unwrapped = reinterpret_cast<SBBreakpointName *>(instance);
  return unwrapped->GetThreadName();
}

void SBBreakpointNameSetQueueName(SBBreakpointNameRef instance,
                                  const char *queue_name) {
  SBBreakpointName *unwrapped = reinterpret_cast<SBBreakpointName *>(instance);
  unwrapped->SetQueueName(queue_name);
}

const char *SBBreakpointNameGetQueueName(SBBreakpointNameRef instance) {
  SBBreakpointName *unwrapped = reinterpret_cast<SBBreakpointName *>(instance);
  return unwrapped->GetQueueName();
}

void SBBreakpointNameSetCallback(SBBreakpointNameRef instance,
                                 lldb::SBBreakpointHitCallback callback,
                                 void *baton) {
  SBBreakpointName *unwrapped = reinterpret_cast<SBBreakpointName *>(instance);
  unwrapped->SetCallback(callback, baton);
}

SBErrorRef
SBBreakpointNameSetScriptCallbackFunction(SBBreakpointNameRef instance,
                                          const char *callback_function_name,
                                          SBStructuredDataRef extra_args) {
  SBBreakpointName *unwrapped = reinterpret_cast<SBBreakpointName *>(instance);
  return reinterpret_cast<SBErrorRef>(
      new SBError(unwrapped->SetScriptCallbackFunction(
          callback_function_name,
          *reinterpret_cast<SBStructuredData *>(extra_args))));
}

void SBBreakpointNameSetCommandLineCommands(SBBreakpointNameRef instance,
                                            SBStringListRef commands) {
  SBBreakpointName *unwrapped = reinterpret_cast<SBBreakpointName *>(instance);
  unwrapped->SetCommandLineCommands(
      *reinterpret_cast<SBStringList *>(commands));
}

bool SBBreakpointNameGetCommandLineCommands(SBBreakpointNameRef instance,
                                            SBStringListRef commands) {
  SBBreakpointName *unwrapped = reinterpret_cast<SBBreakpointName *>(instance);
  return unwrapped->GetCommandLineCommands(
      *reinterpret_cast<SBStringList *>(commands));
}

SBErrorRef SBBreakpointNameSetScriptCallbackBody(SBBreakpointNameRef instance,
                                                 const char *script_body_text) {
  SBBreakpointName *unwrapped = reinterpret_cast<SBBreakpointName *>(instance);
  return reinterpret_cast<SBErrorRef>(
      new SBError(unwrapped->SetScriptCallbackBody(script_body_text)));
}

void SBBreakpointNameSetHelpString(SBBreakpointNameRef instance,
                                   const char *help_string) {
  SBBreakpointName *unwrapped = reinterpret_cast<SBBreakpointName *>(instance);
  unwrapped->SetHelpString(help_string);
}

const char *SBBreakpointNameGetHelpString(SBBreakpointNameRef instance) {
  SBBreakpointName *unwrapped = reinterpret_cast<SBBreakpointName *>(instance);
  return unwrapped->GetHelpString();
}

void SBBreakpointNameSetAllowList(SBBreakpointNameRef instance, bool value) {
  SBBreakpointName *unwrapped = reinterpret_cast<SBBreakpointName *>(instance);
  unwrapped->SetAllowList(value);
}

bool SBBreakpointNameGetAllowList(SBBreakpointNameRef instance) {
  SBBreakpointName *unwrapped = reinterpret_cast<SBBreakpointName *>(instance);
  return unwrapped->GetAllowList();
}

void SBBreakpointNameSetAllowDelete(SBBreakpointNameRef instance, bool value) {
  SBBreakpointName *unwrapped = reinterpret_cast<SBBreakpointName *>(instance);
  unwrapped->SetAllowDelete(value);
}

bool SBBreakpointNameGetAllowDelete(SBBreakpointNameRef instance) {
  SBBreakpointName *unwrapped = reinterpret_cast<SBBreakpointName *>(instance);
  return unwrapped->GetAllowDelete();
}

void SBBreakpointNameSetAllowDisable(SBBreakpointNameRef instance, bool value) {
  SBBreakpointName *unwrapped = reinterpret_cast<SBBreakpointName *>(instance);
  unwrapped->SetAllowDisable(value);
}

bool SBBreakpointNameGetAllowDisable(SBBreakpointNameRef instance) {
  SBBreakpointName *unwrapped = reinterpret_cast<SBBreakpointName *>(instance);
  return unwrapped->GetAllowDisable();
}

bool SBBreakpointNameGetDescription(SBBreakpointNameRef instance,
                                    SBStreamRef description) {
  SBBreakpointName *unwrapped = reinterpret_cast<SBBreakpointName *>(instance);
  return unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description));
}

#ifdef __cplusplus
}
#endif
