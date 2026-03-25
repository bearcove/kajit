//===-- SBTypeSummaryBinding.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/LLDB.h"
#include "lldb/API/SBTypeSummary.h"
#include "lldb/Bindings/LLDBBinding.h"

using namespace lldb;

#ifdef __cplusplus
extern "C" {
#endif

SBTypeSummaryOptionsRef CreateSBTypeSummaryOptions() {
  return reinterpret_cast<SBTypeSummaryOptionsRef>(new SBTypeSummaryOptions());
}

SBTypeSummaryOptionsRef
CloneSBTypeSummaryOptions(SBTypeSummaryOptionsRef instance) {
  return reinterpret_cast<SBTypeSummaryOptionsRef>(new SBTypeSummaryOptions(
      *reinterpret_cast<SBTypeSummaryOptions *>(instance)));
}

void DisposeSBTypeSummaryOptions(SBTypeSummaryOptionsRef instance) {
  delete reinterpret_cast<SBTypeSummaryOptions *>(instance);
}

bool SBTypeSummaryOptionsIsValid(SBTypeSummaryOptionsRef instance) {
  SBTypeSummaryOptions *unwrapped =
      reinterpret_cast<SBTypeSummaryOptions *>(instance);
  return unwrapped->IsValid();
}

enum lldb::LanguageType
SBTypeSummaryOptionsGetLanguage(SBTypeSummaryOptionsRef instance) {
  SBTypeSummaryOptions *unwrapped =
      reinterpret_cast<SBTypeSummaryOptions *>(instance);
  return unwrapped->GetLanguage();
}

enum lldb::TypeSummaryCapping
SBTypeSummaryOptionsGetCapping(SBTypeSummaryOptionsRef instance) {
  SBTypeSummaryOptions *unwrapped =
      reinterpret_cast<SBTypeSummaryOptions *>(instance);
  return unwrapped->GetCapping();
}

void SBTypeSummaryOptionsSetLanguage(SBTypeSummaryOptionsRef instance,
                                     enum lldb::LanguageType var0) {
  SBTypeSummaryOptions *unwrapped =
      reinterpret_cast<SBTypeSummaryOptions *>(instance);
  unwrapped->SetLanguage(var0);
}

void SBTypeSummaryOptionsSetCapping(SBTypeSummaryOptionsRef instance,
                                    enum lldb::TypeSummaryCapping var0) {
  SBTypeSummaryOptions *unwrapped =
      reinterpret_cast<SBTypeSummaryOptions *>(instance);
  unwrapped->SetCapping(var0);
}

SBTypeSummaryRef CreateSBTypeSummary() {
  return reinterpret_cast<SBTypeSummaryRef>(new SBTypeSummary());
}

SBTypeSummaryRef SBTypeSummaryCreateWithSummaryString(const char *data,
                                                      uint32_t options) {
  return reinterpret_cast<SBTypeSummaryRef>(new SBTypeSummary(
      lldb::SBTypeSummary::CreateWithSummaryString(data, options)));
}

SBTypeSummaryRef SBTypeSummaryCreateWithFunctionName(const char *data,
                                                     uint32_t options) {
  return reinterpret_cast<SBTypeSummaryRef>(new SBTypeSummary(
      lldb::SBTypeSummary::CreateWithFunctionName(data, options)));
}

SBTypeSummaryRef SBTypeSummaryCreateWithScriptCode(const char *data,
                                                   uint32_t options) {
  return reinterpret_cast<SBTypeSummaryRef>(new SBTypeSummary(
      lldb::SBTypeSummary::CreateWithScriptCode(data, options)));
}

SBTypeSummaryRef CloneSBTypeSummary(SBTypeSummaryRef instance) {
  return reinterpret_cast<SBTypeSummaryRef>(
      new SBTypeSummary(*reinterpret_cast<SBTypeSummary *>(instance)));
}

void DisposeSBTypeSummary(SBTypeSummaryRef instance) {
  delete reinterpret_cast<SBTypeSummary *>(instance);
}

bool SBTypeSummaryIsValid(SBTypeSummaryRef instance) {
  SBTypeSummary *unwrapped = reinterpret_cast<SBTypeSummary *>(instance);
  return unwrapped->IsValid();
}

bool SBTypeSummaryIsFunctionCode(SBTypeSummaryRef instance) {
  SBTypeSummary *unwrapped = reinterpret_cast<SBTypeSummary *>(instance);
  return unwrapped->IsFunctionCode();
}

bool SBTypeSummaryIsFunctionName(SBTypeSummaryRef instance) {
  SBTypeSummary *unwrapped = reinterpret_cast<SBTypeSummary *>(instance);
  return unwrapped->IsFunctionName();
}

bool SBTypeSummaryIsSummaryString(SBTypeSummaryRef instance) {
  SBTypeSummary *unwrapped = reinterpret_cast<SBTypeSummary *>(instance);
  return unwrapped->IsSummaryString();
}

const char *SBTypeSummaryGetData(SBTypeSummaryRef instance) {
  SBTypeSummary *unwrapped = reinterpret_cast<SBTypeSummary *>(instance);
  return unwrapped->GetData();
}

void SBTypeSummarySetSummaryString(SBTypeSummaryRef instance,
                                   const char *data) {
  SBTypeSummary *unwrapped = reinterpret_cast<SBTypeSummary *>(instance);
  unwrapped->SetSummaryString(data);
}

void SBTypeSummarySetFunctionName(SBTypeSummaryRef instance, const char *data) {
  SBTypeSummary *unwrapped = reinterpret_cast<SBTypeSummary *>(instance);
  unwrapped->SetFunctionName(data);
}

void SBTypeSummarySetFunctionCode(SBTypeSummaryRef instance, const char *data) {
  SBTypeSummary *unwrapped = reinterpret_cast<SBTypeSummary *>(instance);
  unwrapped->SetFunctionCode(data);
}

uint32_t SBTypeSummaryGetOptions(SBTypeSummaryRef instance) {
  SBTypeSummary *unwrapped = reinterpret_cast<SBTypeSummary *>(instance);
  return unwrapped->GetOptions();
}

void SBTypeSummarySetOptions(SBTypeSummaryRef instance, uint32_t var0) {
  SBTypeSummary *unwrapped = reinterpret_cast<SBTypeSummary *>(instance);
  unwrapped->SetOptions(var0);
}

bool SBTypeSummaryGetDescription(SBTypeSummaryRef instance,
                                 SBStreamRef description,
                                 lldb::DescriptionLevel description_level) {
  SBTypeSummary *unwrapped = reinterpret_cast<SBTypeSummary *>(instance);
  return unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description),
                                   description_level);
}

bool SBTypeSummaryIsEqualTo(SBTypeSummaryRef instance, SBTypeSummaryRef rhs) {
  SBTypeSummary *unwrapped = reinterpret_cast<SBTypeSummary *>(instance);
  return unwrapped->IsEqualTo(*reinterpret_cast<SBTypeSummary *>(rhs));
}

#ifdef __cplusplus
}
#endif
