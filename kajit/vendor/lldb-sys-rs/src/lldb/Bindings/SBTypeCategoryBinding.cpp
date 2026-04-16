//===-- SBTypeCategoryBinding.cpp -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/LLDB.h"
#include "lldb/API/SBTypeCategory.h"
#include "lldb/API/SBTypeFilter.h"
#include "lldb/API/SBTypeFormat.h"
#include "lldb/API/SBTypeNameSpecifier.h"
#include "lldb/API/SBTypeSummary.h"
#include "lldb/API/SBTypeSynthetic.h"
#include "lldb/Bindings/LLDBBinding.h"

using namespace lldb;

#ifdef __cplusplus
extern "C" {
#endif

SBTypeCategoryRef CreateSBTypeCategory() {
  return reinterpret_cast<SBTypeCategoryRef>(new SBTypeCategory());
}

SBTypeCategoryRef CloneSBTypeCategory(SBTypeCategoryRef instance) {
  return reinterpret_cast<SBTypeCategoryRef>(
      new SBTypeCategory(*reinterpret_cast<SBTypeCategory *>(instance)));
}

void DisposeSBTypeCategory(SBTypeCategoryRef instance) {
  delete reinterpret_cast<SBTypeCategory *>(instance);
}

bool SBTypeCategoryIsValid(SBTypeCategoryRef instance) {
  SBTypeCategory *unwrapped = reinterpret_cast<SBTypeCategory *>(instance);
  return unwrapped->IsValid();
}

bool SBTypeCategoryGetEnabled(SBTypeCategoryRef instance) {
  SBTypeCategory *unwrapped = reinterpret_cast<SBTypeCategory *>(instance);
  return unwrapped->GetEnabled();
}

void SBTypeCategorySetEnabled(SBTypeCategoryRef instance, bool var0) {
  SBTypeCategory *unwrapped = reinterpret_cast<SBTypeCategory *>(instance);
  unwrapped->SetEnabled(var0);
}

const char *SBTypeCategoryGetName(SBTypeCategoryRef instance) {
  SBTypeCategory *unwrapped = reinterpret_cast<SBTypeCategory *>(instance);
  return unwrapped->GetName();
}

ENUM(LanguageType)
SBTypeCategoryGetLanguageAtIndex(SBTypeCategoryRef instance, uint32_t idx) {
  SBTypeCategory *unwrapped = reinterpret_cast<SBTypeCategory *>(instance);
  return unwrapped->GetLanguageAtIndex(idx);
}

uint32_t SBTypeCategoryGetNumLanguages(SBTypeCategoryRef instance) {
  SBTypeCategory *unwrapped = reinterpret_cast<SBTypeCategory *>(instance);
  return unwrapped->GetNumLanguages();
}

void SBTypeCategoryAddLanguage(SBTypeCategoryRef instance,
                               ENUM(LanguageType) language) {
  SBTypeCategory *unwrapped = reinterpret_cast<SBTypeCategory *>(instance);
  unwrapped->AddLanguage(language);
}

bool SBTypeCategoryGetDescription(SBTypeCategoryRef instance,
                                  SBStreamRef description,
                                  lldb::DescriptionLevel description_level) {
  SBTypeCategory *unwrapped = reinterpret_cast<SBTypeCategory *>(instance);
  return unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description),
                                   description_level);
}

uint32_t SBTypeCategoryGetNumFormats(SBTypeCategoryRef instance) {
  SBTypeCategory *unwrapped = reinterpret_cast<SBTypeCategory *>(instance);
  return unwrapped->GetNumFormats();
}

uint32_t SBTypeCategoryGetNumSummaries(SBTypeCategoryRef instance) {
  SBTypeCategory *unwrapped = reinterpret_cast<SBTypeCategory *>(instance);
  return unwrapped->GetNumSummaries();
}

uint32_t SBTypeCategoryGetNumFilters(SBTypeCategoryRef instance) {
  SBTypeCategory *unwrapped = reinterpret_cast<SBTypeCategory *>(instance);
  return unwrapped->GetNumFilters();
}

uint32_t SBTypeCategoryGetNumSynthetics(SBTypeCategoryRef instance) {
  SBTypeCategory *unwrapped = reinterpret_cast<SBTypeCategory *>(instance);
  return unwrapped->GetNumSynthetics();
}

SBTypeNameSpecifierRef
SBTypeCategoryGetTypeNameSpecifierForFilterAtIndex(SBTypeCategoryRef instance,
                                                   uint32_t var0) {
  SBTypeCategory *unwrapped = reinterpret_cast<SBTypeCategory *>(instance);
  return reinterpret_cast<SBTypeNameSpecifierRef>(new SBTypeNameSpecifier(
      unwrapped->GetTypeNameSpecifierForFilterAtIndex(var0)));
}

SBTypeNameSpecifierRef
SBTypeCategoryGetTypeNameSpecifierForFormatAtIndex(SBTypeCategoryRef instance,
                                                   uint32_t var0) {
  SBTypeCategory *unwrapped = reinterpret_cast<SBTypeCategory *>(instance);
  return reinterpret_cast<SBTypeNameSpecifierRef>(new SBTypeNameSpecifier(
      unwrapped->GetTypeNameSpecifierForFormatAtIndex(var0)));
}

SBTypeNameSpecifierRef
SBTypeCategoryGetTypeNameSpecifierForSummaryAtIndex(SBTypeCategoryRef instance,
                                                    uint32_t var0) {
  SBTypeCategory *unwrapped = reinterpret_cast<SBTypeCategory *>(instance);
  return reinterpret_cast<SBTypeNameSpecifierRef>(new SBTypeNameSpecifier(
      unwrapped->GetTypeNameSpecifierForSummaryAtIndex(var0)));
}

SBTypeNameSpecifierRef SBTypeCategoryGetTypeNameSpecifierForSyntheticAtIndex(
    SBTypeCategoryRef instance, uint32_t var0) {
  SBTypeCategory *unwrapped = reinterpret_cast<SBTypeCategory *>(instance);
  return reinterpret_cast<SBTypeNameSpecifierRef>(new SBTypeNameSpecifier(
      unwrapped->GetTypeNameSpecifierForSyntheticAtIndex(var0)));
}

SBTypeFilterRef SBTypeCategoryGetFilterForType(SBTypeCategoryRef instance,
                                               SBTypeNameSpecifierRef var0) {
  SBTypeCategory *unwrapped = reinterpret_cast<SBTypeCategory *>(instance);
  return reinterpret_cast<SBTypeFilterRef>(
      new SBTypeFilter(unwrapped->GetFilterForType(
          *reinterpret_cast<SBTypeNameSpecifier *>(var0))));
}

SBTypeFormatRef SBTypeCategoryGetFormatForType(SBTypeCategoryRef instance,
                                               SBTypeNameSpecifierRef var0) {
  SBTypeCategory *unwrapped = reinterpret_cast<SBTypeCategory *>(instance);
  return reinterpret_cast<SBTypeFormatRef>(
      new SBTypeFormat(unwrapped->GetFormatForType(
          *reinterpret_cast<SBTypeNameSpecifier *>(var0))));
}

SBTypeSummaryRef SBTypeCategoryGetSummaryForType(SBTypeCategoryRef instance,
                                                 SBTypeNameSpecifierRef var0) {
  SBTypeCategory *unwrapped = reinterpret_cast<SBTypeCategory *>(instance);
  return reinterpret_cast<SBTypeSummaryRef>(
      new SBTypeSummary(unwrapped->GetSummaryForType(
          *reinterpret_cast<SBTypeNameSpecifier *>(var0))));
}

SBTypeSyntheticRef
SBTypeCategoryGetSyntheticForType(SBTypeCategoryRef instance,
                                  SBTypeNameSpecifierRef var0) {
  SBTypeCategory *unwrapped = reinterpret_cast<SBTypeCategory *>(instance);
  return reinterpret_cast<SBTypeSyntheticRef>(
      new SBTypeSynthetic(unwrapped->GetSyntheticForType(
          *reinterpret_cast<SBTypeNameSpecifier *>(var0))));
}

SBTypeFilterRef SBTypeCategoryGetFilterAtIndex(SBTypeCategoryRef instance,
                                               uint32_t var0) {
  SBTypeCategory *unwrapped = reinterpret_cast<SBTypeCategory *>(instance);
  return reinterpret_cast<SBTypeFilterRef>(
      new SBTypeFilter(unwrapped->GetFilterAtIndex(var0)));
}

SBTypeFormatRef SBTypeCategoryGetFormatAtIndex(SBTypeCategoryRef instance,
                                               uint32_t var0) {
  SBTypeCategory *unwrapped = reinterpret_cast<SBTypeCategory *>(instance);
  return reinterpret_cast<SBTypeFormatRef>(
      new SBTypeFormat(unwrapped->GetFormatAtIndex(var0)));
}

SBTypeSummaryRef SBTypeCategoryGetSummaryAtIndex(SBTypeCategoryRef instance,
                                                 uint32_t var0) {
  SBTypeCategory *unwrapped = reinterpret_cast<SBTypeCategory *>(instance);
  return reinterpret_cast<SBTypeSummaryRef>(
      new SBTypeSummary(unwrapped->GetSummaryAtIndex(var0)));
}

SBTypeSyntheticRef SBTypeCategoryGetSyntheticAtIndex(SBTypeCategoryRef instance,
                                                     uint32_t var0) {
  SBTypeCategory *unwrapped = reinterpret_cast<SBTypeCategory *>(instance);
  return reinterpret_cast<SBTypeSyntheticRef>(
      new SBTypeSynthetic(unwrapped->GetSyntheticAtIndex(var0)));
}

bool SBTypeCategoryAddTypeFormat(SBTypeCategoryRef instance,
                                 SBTypeNameSpecifierRef var0,
                                 SBTypeFormatRef var1) {
  SBTypeCategory *unwrapped = reinterpret_cast<SBTypeCategory *>(instance);
  return unwrapped->AddTypeFormat(
      *reinterpret_cast<SBTypeNameSpecifier *>(var0),
      *reinterpret_cast<SBTypeFormat *>(var1));
}

bool SBTypeCategoryDeleteTypeFormat(SBTypeCategoryRef instance,
                                    SBTypeNameSpecifierRef var0) {
  SBTypeCategory *unwrapped = reinterpret_cast<SBTypeCategory *>(instance);
  return unwrapped->DeleteTypeFormat(
      *reinterpret_cast<SBTypeNameSpecifier *>(var0));
}

bool SBTypeCategoryAddTypeSummary(SBTypeCategoryRef instance,
                                  SBTypeNameSpecifierRef var0,
                                  SBTypeSummaryRef var1) {
  SBTypeCategory *unwrapped = reinterpret_cast<SBTypeCategory *>(instance);
  return unwrapped->AddTypeSummary(
      *reinterpret_cast<SBTypeNameSpecifier *>(var0),
      *reinterpret_cast<SBTypeSummary *>(var1));
}

bool SBTypeCategoryDeleteTypeSummary(SBTypeCategoryRef instance,
                                     SBTypeNameSpecifierRef var0) {
  SBTypeCategory *unwrapped = reinterpret_cast<SBTypeCategory *>(instance);
  return unwrapped->DeleteTypeSummary(
      *reinterpret_cast<SBTypeNameSpecifier *>(var0));
}

bool SBTypeCategoryAddTypeFilter(SBTypeCategoryRef instance,
                                 SBTypeNameSpecifierRef var0,
                                 SBTypeFilterRef var1) {
  SBTypeCategory *unwrapped = reinterpret_cast<SBTypeCategory *>(instance);
  return unwrapped->AddTypeFilter(
      *reinterpret_cast<SBTypeNameSpecifier *>(var0),
      *reinterpret_cast<SBTypeFilter *>(var1));
}

bool SBTypeCategoryDeleteTypeFilter(SBTypeCategoryRef instance,
                                    SBTypeNameSpecifierRef var0) {
  SBTypeCategory *unwrapped = reinterpret_cast<SBTypeCategory *>(instance);
  return unwrapped->DeleteTypeFilter(
      *reinterpret_cast<SBTypeNameSpecifier *>(var0));
}

bool SBTypeCategoryAddTypeSynthetic(SBTypeCategoryRef instance,
                                    SBTypeNameSpecifierRef var0,
                                    SBTypeSyntheticRef var1) {
  SBTypeCategory *unwrapped = reinterpret_cast<SBTypeCategory *>(instance);
  return unwrapped->AddTypeSynthetic(
      *reinterpret_cast<SBTypeNameSpecifier *>(var0),
      *reinterpret_cast<SBTypeSynthetic *>(var1));
}

bool SBTypeCategoryDeleteTypeSynthetic(SBTypeCategoryRef instance,
                                       SBTypeNameSpecifierRef var0) {
  SBTypeCategory *unwrapped = reinterpret_cast<SBTypeCategory *>(instance);
  return unwrapped->DeleteTypeSynthetic(
      *reinterpret_cast<SBTypeNameSpecifier *>(var0));
}

#ifdef __cplusplus
}
#endif
