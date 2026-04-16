//===-- SBTypeFilterBinding.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/LLDB.h"
#include "lldb/API/SBTypeFilter.h"
#include "lldb/Bindings/LLDBBinding.h"

using namespace lldb;

#ifdef __cplusplus
extern "C" {
#endif

SBTypeFilterRef CreateSBTypeFilter() {
  return reinterpret_cast<SBTypeFilterRef>(new SBTypeFilter());
}

SBTypeFilterRef CreateSBTypeFilter2(uint32_t options) {
  return reinterpret_cast<SBTypeFilterRef>(new SBTypeFilter(options));
}

SBTypeFilterRef CloneSBTypeFilter(SBTypeFilterRef instance) {
  return reinterpret_cast<SBTypeFilterRef>(
      new SBTypeFilter(*reinterpret_cast<SBTypeFilter *>(instance)));
}

void DisposeSBTypeFilter(SBTypeFilterRef instance) {
  delete reinterpret_cast<SBTypeFilter *>(instance);
}

bool SBTypeFilterIsValid(SBTypeFilterRef instance) {
  SBTypeFilter *unwrapped = reinterpret_cast<SBTypeFilter *>(instance);
  return unwrapped->IsValid();
}

uint32_t SBTypeFilterGetNumberOfExpressionPaths(SBTypeFilterRef instance) {
  SBTypeFilter *unwrapped = reinterpret_cast<SBTypeFilter *>(instance);
  return unwrapped->GetNumberOfExpressionPaths();
}

const char *SBTypeFilterGetExpressionPathAtIndex(SBTypeFilterRef instance,
                                                 uint32_t i) {
  SBTypeFilter *unwrapped = reinterpret_cast<SBTypeFilter *>(instance);
  return unwrapped->GetExpressionPathAtIndex(i);
}

bool SBTypeFilterReplaceExpressionPathAtIndex(SBTypeFilterRef instance,
                                              uint32_t i, const char *item) {
  SBTypeFilter *unwrapped = reinterpret_cast<SBTypeFilter *>(instance);
  return unwrapped->ReplaceExpressionPathAtIndex(i, item);
}

void SBTypeFilterAppendExpressionPath(SBTypeFilterRef instance,
                                      const char *item) {
  SBTypeFilter *unwrapped = reinterpret_cast<SBTypeFilter *>(instance);
  unwrapped->AppendExpressionPath(item);
}

void SBTypeFilterClear(SBTypeFilterRef instance) {
  SBTypeFilter *unwrapped = reinterpret_cast<SBTypeFilter *>(instance);
  unwrapped->Clear();
}

uint32_t SBTypeFilterGetOptions(SBTypeFilterRef instance) {
  SBTypeFilter *unwrapped = reinterpret_cast<SBTypeFilter *>(instance);
  return unwrapped->GetOptions();
}

void SBTypeFilterSetOptions(SBTypeFilterRef instance, uint32_t var0) {
  SBTypeFilter *unwrapped = reinterpret_cast<SBTypeFilter *>(instance);
  unwrapped->SetOptions(var0);
}

bool SBTypeFilterGetDescription(SBTypeFilterRef instance,
                                SBStreamRef description,
                                lldb::DescriptionLevel description_level) {
  SBTypeFilter *unwrapped = reinterpret_cast<SBTypeFilter *>(instance);
  return unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description),
                                   description_level);
}

bool SBTypeFilterIsEqualTo(SBTypeFilterRef instance, SBTypeFilterRef rhs) {
  SBTypeFilter *unwrapped = reinterpret_cast<SBTypeFilter *>(instance);
  return unwrapped->IsEqualTo(*reinterpret_cast<SBTypeFilter *>(rhs));
}

#ifdef __cplusplus
}
#endif
