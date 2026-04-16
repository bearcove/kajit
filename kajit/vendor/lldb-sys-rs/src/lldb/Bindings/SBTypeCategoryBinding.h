//===-- SBTypeCategoryBinding.h ----------------------------------*- C++
//-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBTypeCategoryBinding_h_
#define LLDB_SBTypeCategoryBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBTypeCategoryRef CreateSBTypeCategory(void);

LLDB_API SBTypeCategoryRef CloneSBTypeCategory(SBTypeCategoryRef instance);

LLDB_API void DisposeSBTypeCategory(SBTypeCategoryRef instance);

LLDB_API bool SBTypeCategoryIsValid(SBTypeCategoryRef instance);

LLDB_API bool SBTypeCategoryGetEnabled(SBTypeCategoryRef instance);

LLDB_API void SBTypeCategorySetEnabled(SBTypeCategoryRef instance, bool);

LLDB_API const char *SBTypeCategoryGetName(SBTypeCategoryRef instance);

LLDB_API ENUM(LanguageType)
    SBTypeCategoryGetLanguageAtIndex(SBTypeCategoryRef instance, uint32_t idx);

LLDB_API uint32_t SBTypeCategoryGetNumLanguages(SBTypeCategoryRef instance);

LLDB_API void SBTypeCategoryAddLanguage(SBTypeCategoryRef instance,
                                        ENUM(LanguageType) language);

LLDB_API bool SBTypeCategoryGetDescription(SBTypeCategoryRef instance,
                                           SBStreamRef description,
                                           ENUM(DescriptionLevel)
                                               description_level);

LLDB_API uint32_t SBTypeCategoryGetNumFormats(SBTypeCategoryRef instance);

LLDB_API uint32_t SBTypeCategoryGetNumSummaries(SBTypeCategoryRef instance);

LLDB_API uint32_t SBTypeCategoryGetNumFilters(SBTypeCategoryRef instance);

LLDB_API uint32_t SBTypeCategoryGetNumSynthetics(SBTypeCategoryRef instance);

LLDB_API SBTypeNameSpecifierRef
SBTypeCategoryGetTypeNameSpecifierForFilterAtIndex(SBTypeCategoryRef instance,
                                                   uint32_t);

LLDB_API SBTypeNameSpecifierRef
SBTypeCategoryGetTypeNameSpecifierForFormatAtIndex(SBTypeCategoryRef instance,
                                                   uint32_t);

LLDB_API SBTypeNameSpecifierRef
SBTypeCategoryGetTypeNameSpecifierForSummaryAtIndex(SBTypeCategoryRef instance,
                                                    uint32_t);

LLDB_API SBTypeNameSpecifierRef
SBTypeCategoryGetTypeNameSpecifierForSyntheticAtIndex(
    SBTypeCategoryRef instance, uint32_t);

LLDB_API SBTypeFilterRef SBTypeCategoryGetFilterForType(
    SBTypeCategoryRef instance, SBTypeNameSpecifierRef);

LLDB_API SBTypeFormatRef SBTypeCategoryGetFormatForType(
    SBTypeCategoryRef instance, SBTypeNameSpecifierRef);

LLDB_API SBTypeSummaryRef SBTypeCategoryGetSummaryForType(
    SBTypeCategoryRef instance, SBTypeNameSpecifierRef);

LLDB_API SBTypeSyntheticRef SBTypeCategoryGetSyntheticForType(
    SBTypeCategoryRef instance, SBTypeNameSpecifierRef);

LLDB_API SBTypeFilterRef
SBTypeCategoryGetFilterAtIndex(SBTypeCategoryRef instance, uint32_t);

LLDB_API SBTypeFormatRef
SBTypeCategoryGetFormatAtIndex(SBTypeCategoryRef instance, uint32_t);

LLDB_API SBTypeSummaryRef
SBTypeCategoryGetSummaryAtIndex(SBTypeCategoryRef instance, uint32_t);

LLDB_API SBTypeSyntheticRef
SBTypeCategoryGetSyntheticAtIndex(SBTypeCategoryRef instance, uint32_t);

LLDB_API bool SBTypeCategoryAddTypeFormat(SBTypeCategoryRef instance,
                                          SBTypeNameSpecifierRef,
                                          SBTypeFormatRef);

LLDB_API bool SBTypeCategoryDeleteTypeFormat(SBTypeCategoryRef instance,
                                             SBTypeNameSpecifierRef);

LLDB_API bool SBTypeCategoryAddTypeSummary(SBTypeCategoryRef instance,
                                           SBTypeNameSpecifierRef,
                                           SBTypeSummaryRef);

LLDB_API bool SBTypeCategoryDeleteTypeSummary(SBTypeCategoryRef instance,
                                              SBTypeNameSpecifierRef);

LLDB_API bool SBTypeCategoryAddTypeFilter(SBTypeCategoryRef instance,
                                          SBTypeNameSpecifierRef,
                                          SBTypeFilterRef);

LLDB_API bool SBTypeCategoryDeleteTypeFilter(SBTypeCategoryRef instance,
                                             SBTypeNameSpecifierRef);

LLDB_API bool SBTypeCategoryAddTypeSynthetic(SBTypeCategoryRef instance,
                                             SBTypeNameSpecifierRef,
                                             SBTypeSyntheticRef);

LLDB_API bool SBTypeCategoryDeleteTypeSynthetic(SBTypeCategoryRef instance,
                                                SBTypeNameSpecifierRef);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBTypeCategoryBinding_h_
