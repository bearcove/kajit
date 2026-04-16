//===-- SBVariablesOptionsBinding.cpp ---------------------------*- C++ -*-===//
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

SBVariablesOptionsRef CreateSBVariablesOptions() {
  return reinterpret_cast<SBVariablesOptionsRef>(new SBVariablesOptions());
}

SBVariablesOptionsRef CreateSBVariablesOptions2(SBVariablesOptionsRef options) {
  return reinterpret_cast<SBVariablesOptionsRef>(
      new SBVariablesOptions(*reinterpret_cast<SBVariablesOptions *>(options)));
}

SBVariablesOptionsRef CloneSBVariablesOptions(SBVariablesOptionsRef instance) {
  return reinterpret_cast<SBVariablesOptionsRef>(new SBVariablesOptions(
      *reinterpret_cast<SBVariablesOptions *>(instance)));
}

void DisposeSBVariablesOptions(SBVariablesOptionsRef instance) {
  delete reinterpret_cast<SBVariablesOptions *>(instance);
}

bool SBVariablesOptionsIsValid(SBVariablesOptionsRef instance) {
  SBVariablesOptions *unwrapped =
      reinterpret_cast<SBVariablesOptions *>(instance);
  return unwrapped->IsValid();
}

bool SBVariablesOptionsGetIncludeArguments(SBVariablesOptionsRef instance) {
  SBVariablesOptions *unwrapped =
      reinterpret_cast<SBVariablesOptions *>(instance);
  return unwrapped->GetIncludeArguments();
}

void SBVariablesOptionsSetIncludeArguments(SBVariablesOptionsRef instance,
                                           bool var0) {
  SBVariablesOptions *unwrapped =
      reinterpret_cast<SBVariablesOptions *>(instance);
  unwrapped->SetIncludeArguments(var0);
}

bool SBVariablesOptionsGetIncludeRecognizedArguments(
    SBVariablesOptionsRef instance, SBTargetRef target) {
  SBVariablesOptions *unwrapped =
      reinterpret_cast<SBVariablesOptions *>(instance);
  return unwrapped->GetIncludeRecognizedArguments(
      *reinterpret_cast<SBTarget *>(target));
}

void SBVariablesOptionsSetIncludeRecognizedArguments(
    SBVariablesOptionsRef instance, bool var0) {
  SBVariablesOptions *unwrapped =
      reinterpret_cast<SBVariablesOptions *>(instance);
  unwrapped->SetIncludeRecognizedArguments(var0);
}

bool SBVariablesOptionsGetIncludeLocals(SBVariablesOptionsRef instance) {
  SBVariablesOptions *unwrapped =
      reinterpret_cast<SBVariablesOptions *>(instance);
  return unwrapped->GetIncludeLocals();
}

void SBVariablesOptionsSetIncludeLocals(SBVariablesOptionsRef instance,
                                        bool var0) {
  SBVariablesOptions *unwrapped =
      reinterpret_cast<SBVariablesOptions *>(instance);
  unwrapped->SetIncludeLocals(var0);
}

bool SBVariablesOptionsGetIncludeStatics(SBVariablesOptionsRef instance) {
  SBVariablesOptions *unwrapped =
      reinterpret_cast<SBVariablesOptions *>(instance);
  return unwrapped->GetIncludeStatics();
}

void SBVariablesOptionsSetIncludeStatics(SBVariablesOptionsRef instance,
                                         bool var0) {
  SBVariablesOptions *unwrapped =
      reinterpret_cast<SBVariablesOptions *>(instance);
  unwrapped->SetIncludeStatics(var0);
}

bool SBVariablesOptionsGetInScopeOnly(SBVariablesOptionsRef instance) {
  SBVariablesOptions *unwrapped =
      reinterpret_cast<SBVariablesOptions *>(instance);
  return unwrapped->GetInScopeOnly();
}

void SBVariablesOptionsSetInScopeOnly(SBVariablesOptionsRef instance,
                                      bool var0) {
  SBVariablesOptions *unwrapped =
      reinterpret_cast<SBVariablesOptions *>(instance);
  unwrapped->SetInScopeOnly(var0);
}

bool SBVariablesOptionsGetIncludeRuntimeSupportValues(
    SBVariablesOptionsRef instance) {
  SBVariablesOptions *unwrapped =
      reinterpret_cast<SBVariablesOptions *>(instance);
  return unwrapped->GetIncludeRuntimeSupportValues();
}

void SBVariablesOptionsSetIncludeRuntimeSupportValues(
    SBVariablesOptionsRef instance, bool var0) {
  SBVariablesOptions *unwrapped =
      reinterpret_cast<SBVariablesOptions *>(instance);
  unwrapped->SetIncludeRuntimeSupportValues(var0);
}

enum lldb::DynamicValueType
SBVariablesOptionsGetUseDynamic(SBVariablesOptionsRef instance) {
  SBVariablesOptions *unwrapped =
      reinterpret_cast<SBVariablesOptions *>(instance);
  return unwrapped->GetUseDynamic();
}

void SBVariablesOptionsSetUseDynamic(SBVariablesOptionsRef instance,
                                     lldb::DynamicValueType var0) {
  SBVariablesOptions *unwrapped =
      reinterpret_cast<SBVariablesOptions *>(instance);
  unwrapped->SetUseDynamic(var0);
}

#ifdef __cplusplus
}
#endif
