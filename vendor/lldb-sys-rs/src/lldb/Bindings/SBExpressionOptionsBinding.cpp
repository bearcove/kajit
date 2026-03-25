//===-- SBExpressionOptionsBinding.cpp --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/LLDB.h"
#include "lldb/API/SBExpressionOptions.h"
#include "lldb/Bindings/LLDBBinding.h"

using namespace lldb;

#ifdef __cplusplus
extern "C" {
#endif

SBExpressionOptionsRef CreateSBExpressionOptions() {
  return reinterpret_cast<SBExpressionOptionsRef>(new SBExpressionOptions());
}

SBExpressionOptionsRef
CloneSBExpressionOptions(SBExpressionOptionsRef instance) {
  return reinterpret_cast<SBExpressionOptionsRef>(new SBExpressionOptions(
      *reinterpret_cast<SBExpressionOptions *>(instance)));
}

void DisposeSBExpressionOptions(SBExpressionOptionsRef instance) {
  delete reinterpret_cast<SBExpressionOptions *>(instance);
}

bool SBExpressionOptionsGetCoerceResultToId(SBExpressionOptionsRef instance) {
  SBExpressionOptions *unwrapped =
      reinterpret_cast<SBExpressionOptions *>(instance);
  return unwrapped->GetCoerceResultToId();
}

void SBExpressionOptionsSetCoerceResultToId(SBExpressionOptionsRef instance,
                                            bool coerce) {
  SBExpressionOptions *unwrapped =
      reinterpret_cast<SBExpressionOptions *>(instance);
  unwrapped->SetCoerceResultToId(coerce);
}

bool SBExpressionOptionsGetUnwindOnError(SBExpressionOptionsRef instance) {
  SBExpressionOptions *unwrapped =
      reinterpret_cast<SBExpressionOptions *>(instance);
  return unwrapped->GetUnwindOnError();
}

void SBExpressionOptionsSetUnwindOnError(SBExpressionOptionsRef instance,
                                         bool unwind) {
  SBExpressionOptions *unwrapped =
      reinterpret_cast<SBExpressionOptions *>(instance);
  unwrapped->SetUnwindOnError(unwind);
}

bool SBExpressionOptionsGetIgnoreBreakpoints(SBExpressionOptionsRef instance) {
  SBExpressionOptions *unwrapped =
      reinterpret_cast<SBExpressionOptions *>(instance);
  return unwrapped->GetIgnoreBreakpoints();
}

void SBExpressionOptionsSetIgnoreBreakpoints(SBExpressionOptionsRef instance,
                                             bool ignore) {
  SBExpressionOptions *unwrapped =
      reinterpret_cast<SBExpressionOptions *>(instance);
  unwrapped->SetIgnoreBreakpoints(ignore);
}

enum lldb::DynamicValueType
SBExpressionOptionsGetFetchDynamicValue(SBExpressionOptionsRef instance) {
  SBExpressionOptions *unwrapped =
      reinterpret_cast<SBExpressionOptions *>(instance);
  return unwrapped->GetFetchDynamicValue();
}

void SBExpressionOptionsSetFetchDynamicValue(SBExpressionOptionsRef instance,
                                             lldb::DynamicValueType dynamic) {
  SBExpressionOptions *unwrapped =
      reinterpret_cast<SBExpressionOptions *>(instance);
  unwrapped->SetFetchDynamicValue(dynamic);
}

uint32_t
SBExpressionOptionsGetTimeoutInMicroSeconds(SBExpressionOptionsRef instance) {
  SBExpressionOptions *unwrapped =
      reinterpret_cast<SBExpressionOptions *>(instance);
  return unwrapped->GetTimeoutInMicroSeconds();
}

void SBExpressionOptionsSetTimeoutInMicroSeconds(
    SBExpressionOptionsRef instance, uint32_t timeout) {
  SBExpressionOptions *unwrapped =
      reinterpret_cast<SBExpressionOptions *>(instance);
  unwrapped->SetTimeoutInMicroSeconds(timeout);
}

uint32_t SBExpressionOptionsGetOneThreadTimeoutInMicroSeconds(
    SBExpressionOptionsRef instance) {
  SBExpressionOptions *unwrapped =
      reinterpret_cast<SBExpressionOptions *>(instance);
  return unwrapped->GetOneThreadTimeoutInMicroSeconds();
}

void SBExpressionOptionsSetOneThreadTimeoutInMicroSeconds(
    SBExpressionOptionsRef instance, uint32_t timeout) {
  SBExpressionOptions *unwrapped =
      reinterpret_cast<SBExpressionOptions *>(instance);
  unwrapped->SetOneThreadTimeoutInMicroSeconds(timeout);
}

bool SBExpressionOptionsGetTryAllThreads(SBExpressionOptionsRef instance) {
  SBExpressionOptions *unwrapped =
      reinterpret_cast<SBExpressionOptions *>(instance);
  return unwrapped->GetTryAllThreads();
}

void SBExpressionOptionsSetTryAllThreads(SBExpressionOptionsRef instance,
                                         bool run_others) {
  SBExpressionOptions *unwrapped =
      reinterpret_cast<SBExpressionOptions *>(instance);
  unwrapped->SetTryAllThreads(run_others);
}

bool SBExpressionOptionsGetStopOthers(SBExpressionOptionsRef instance) {
  SBExpressionOptions *unwrapped =
      reinterpret_cast<SBExpressionOptions *>(instance);
  return unwrapped->GetStopOthers();
}

void SBExpressionOptionsSetStopOthers(SBExpressionOptionsRef instance,
                                      bool stop_others) {
  SBExpressionOptions *unwrapped =
      reinterpret_cast<SBExpressionOptions *>(instance);
  unwrapped->SetStopOthers(stop_others);
}

bool SBExpressionOptionsGetTrapExceptions(SBExpressionOptionsRef instance) {
  SBExpressionOptions *unwrapped =
      reinterpret_cast<SBExpressionOptions *>(instance);
  return unwrapped->GetTrapExceptions();
}

void SBExpressionOptionsSetTrapExceptions(SBExpressionOptionsRef instance,
                                          bool trap_exceptions) {
  SBExpressionOptions *unwrapped =
      reinterpret_cast<SBExpressionOptions *>(instance);
  unwrapped->SetTrapExceptions(trap_exceptions);
}

void SBExpressionOptionsSetLanguage(SBExpressionOptionsRef instance,
                                    lldb::LanguageType language) {
  SBExpressionOptions *unwrapped =
      reinterpret_cast<SBExpressionOptions *>(instance);
  unwrapped->SetLanguage(language);
}

void SBExpressionOptionsSetCancelCallback(
    SBExpressionOptionsRef instance, lldb::ExpressionCancelCallback callback,
    void *baton) {
  SBExpressionOptions *unwrapped =
      reinterpret_cast<SBExpressionOptions *>(instance);
  unwrapped->SetCancelCallback(callback, baton);
}

bool SBExpressionOptionsGetGenerateDebugInfo(SBExpressionOptionsRef instance) {
  SBExpressionOptions *unwrapped =
      reinterpret_cast<SBExpressionOptions *>(instance);
  return unwrapped->GetGenerateDebugInfo();
}

void SBExpressionOptionsSetGenerateDebugInfo(SBExpressionOptionsRef instance,
                                             bool b) {
  SBExpressionOptions *unwrapped =
      reinterpret_cast<SBExpressionOptions *>(instance);
  unwrapped->SetGenerateDebugInfo(b);
}

bool SBExpressionOptionsGetSuppressPersistentResult(
    SBExpressionOptionsRef instance) {
  SBExpressionOptions *unwrapped =
      reinterpret_cast<SBExpressionOptions *>(instance);
  return unwrapped->GetSuppressPersistentResult();
}

void SBExpressionOptionsSetSuppressPersistentResult(
    SBExpressionOptionsRef instance, bool b) {
  SBExpressionOptions *unwrapped =
      reinterpret_cast<SBExpressionOptions *>(instance);
  unwrapped->SetSuppressPersistentResult(b);
}

const char *SBExpressionOptionsGetPrefix(SBExpressionOptionsRef instance) {
  SBExpressionOptions *unwrapped =
      reinterpret_cast<SBExpressionOptions *>(instance);
  return unwrapped->GetPrefix();
}

void SBExpressionOptionsSetPrefix(SBExpressionOptionsRef instance,
                                  const char *prefix) {
  SBExpressionOptions *unwrapped =
      reinterpret_cast<SBExpressionOptions *>(instance);
  unwrapped->SetPrefix(prefix);
}

void SBExpressionOptionsSetAutoApplyFixIts(SBExpressionOptionsRef instance,
                                           bool b) {
  SBExpressionOptions *unwrapped =
      reinterpret_cast<SBExpressionOptions *>(instance);
  unwrapped->SetAutoApplyFixIts(b);
}

bool SBExpressionOptionsGetAutoApplyFixIts(SBExpressionOptionsRef instance) {
  SBExpressionOptions *unwrapped =
      reinterpret_cast<SBExpressionOptions *>(instance);
  return unwrapped->GetAutoApplyFixIts();
}

uint64_t
SBExpressionOptionsGetRetriesWithFixIts(SBExpressionOptionsRef instance) {
  SBExpressionOptions *unwrapped =
      reinterpret_cast<SBExpressionOptions *>(instance);
  return unwrapped->GetRetriesWithFixIts();
}

void SBExpressionOptionsSetRetriesWithFixIts(SBExpressionOptionsRef instance,
                                             uint64_t retries) {
  SBExpressionOptions *unwrapped =
      reinterpret_cast<SBExpressionOptions *>(instance);
  unwrapped->SetRetriesWithFixIts(retries);
}

bool SBExpressionOptionsGetTopLevel(SBExpressionOptionsRef instance) {
  SBExpressionOptions *unwrapped =
      reinterpret_cast<SBExpressionOptions *>(instance);
  return unwrapped->GetTopLevel();
}

void SBExpressionOptionsSetTopLevel(SBExpressionOptionsRef instance, bool b) {
  SBExpressionOptions *unwrapped =
      reinterpret_cast<SBExpressionOptions *>(instance);
  unwrapped->SetTopLevel(b);
}

bool SBExpressionOptionsGetAllowJIT(SBExpressionOptionsRef instance) {
  SBExpressionOptions *unwrapped =
      reinterpret_cast<SBExpressionOptions *>(instance);
  return unwrapped->GetAllowJIT();
}

void SBExpressionOptionsSetAllowJIT(SBExpressionOptionsRef instance, bool b) {
  SBExpressionOptions *unwrapped =
      reinterpret_cast<SBExpressionOptions *>(instance);
  unwrapped->SetAllowJIT(b);
}

#ifdef __cplusplus
}
#endif
