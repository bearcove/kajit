//===-- SBStructuredDataBinding.cpp ---------------------------------------*-
// C++ -*-===//
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

SBStructuredDataRef CreateSBStructuredData() {
  return reinterpret_cast<SBStructuredDataRef>(new SBStructuredData());
}

SBStructuredDataRef CloneSBStructuredData(SBStructuredDataRef instance) {
  return reinterpret_cast<SBStructuredDataRef>(
      new SBStructuredData(*reinterpret_cast<SBStructuredData *>(instance)));
}

void DisposeSBStructuredData(SBStructuredDataRef instance) {
  delete reinterpret_cast<SBStructuredData *>(instance);
}

bool SBStructuredDataIsValid(SBStructuredDataRef instance) {
  SBStructuredData *unwrapped = reinterpret_cast<SBStructuredData *>(instance);
  return unwrapped->IsValid();
}

void SBStructuredDataClear(SBStructuredDataRef instance) {
  SBStructuredData *unwrapped = reinterpret_cast<SBStructuredData *>(instance);
  unwrapped->Clear();
}

SBErrorRef SBStructuredDataSetFromJSON(SBStructuredDataRef instance,
                                       SBStreamRef stream) {
  SBStructuredData *unwrapped = reinterpret_cast<SBStructuredData *>(instance);
  return reinterpret_cast<SBErrorRef>(new SBError(
      unwrapped->SetFromJSON(*reinterpret_cast<SBStream *>(stream))));
}

SBErrorRef SBStructuredDataSetFromJSON2(SBStructuredDataRef instance,
                                        const char *json) {
  SBStructuredData *unwrapped = reinterpret_cast<SBStructuredData *>(instance);
  return reinterpret_cast<SBErrorRef>(
      new SBError(unwrapped->SetFromJSON(json)));
}

SBErrorRef SBStructuredDataGetAsJSON(SBStructuredDataRef instance,
                                     SBStreamRef stream) {
  SBStructuredData *unwrapped = reinterpret_cast<SBStructuredData *>(instance);
  return reinterpret_cast<SBErrorRef>(
      new SBError(unwrapped->GetAsJSON(*reinterpret_cast<SBStream *>(stream))));
}

SBErrorRef SBStructuredDataGetDescription(SBStructuredDataRef instance,
                                          SBStreamRef description) {
  SBStructuredData *unwrapped = reinterpret_cast<SBStructuredData *>(instance);
  return reinterpret_cast<SBErrorRef>(new SBError(
      unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description))));
}

StructuredDataType SBStructuredDataGetType(SBStructuredDataRef instance) {
  SBStructuredData *unwrapped = reinterpret_cast<SBStructuredData *>(instance);
  return unwrapped->GetType();
}

size_t SBStructuredDataGetSize(SBStructuredDataRef instance) {
  SBStructuredData *unwrapped = reinterpret_cast<SBStructuredData *>(instance);
  return unwrapped->GetSize();
}

bool SBStructuredDataGetKeys(SBStructuredDataRef instance,
                             SBStringListRef keys) {
  SBStructuredData *unwrapped = reinterpret_cast<SBStructuredData *>(instance);
  return unwrapped->GetKeys(*reinterpret_cast<SBStringList *>(keys));
}

SBStructuredDataRef SBStructuredDataGetValueForKey(SBStructuredDataRef instance,
                                                   const char *key) {
  SBStructuredData *unwrapped = reinterpret_cast<SBStructuredData *>(instance);
  return reinterpret_cast<SBStructuredDataRef>(
      new SBStructuredData(unwrapped->GetValueForKey(key)));
}

SBStructuredDataRef SBStructuredDataGetItemAtIndex(SBStructuredDataRef instance,
                                                   size_t idx) {
  SBStructuredData *unwrapped = reinterpret_cast<SBStructuredData *>(instance);
  return reinterpret_cast<SBStructuredDataRef>(
      new SBStructuredData(unwrapped->GetItemAtIndex(idx)));
}

uint64_t SBStructuredDataGetIntegerValue(SBStructuredDataRef instance,
                                         uint64_t fail_value) {
  SBStructuredData *unwrapped = reinterpret_cast<SBStructuredData *>(instance);
  return unwrapped->GetIntegerValue(fail_value);
}

double SBStructuredDataGetFloatValue(SBStructuredDataRef instance,
                                     double fail_value) {
  SBStructuredData *unwrapped = reinterpret_cast<SBStructuredData *>(instance);
  return unwrapped->GetFloatValue(fail_value);
}

bool SBStructuredDataGetBooleanValue(SBStructuredDataRef instance,
                                     bool fail_value) {
  SBStructuredData *unwrapped = reinterpret_cast<SBStructuredData *>(instance);
  return unwrapped->GetBooleanValue(fail_value);
}

size_t SBStructuredDataGetStringValue(SBStructuredDataRef instance, char *dst,
                                      size_t dst_len) {
  SBStructuredData *unwrapped = reinterpret_cast<SBStructuredData *>(instance);
  return unwrapped->GetStringValue(dst, dst_len);
}

#ifdef __cplusplus
}
#endif
