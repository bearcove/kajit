//===-- SBTypeSummaryBinding.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBTypeSummaryBinding_h_
#define LLDB_SBTypeSummaryBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBTypeSummaryOptionsRef CreateSBTypeSummaryOptions(void);

LLDB_API SBTypeSummaryOptionsRef
CloneSBTypeSummaryOptions(SBTypeSummaryOptionsRef instance);

LLDB_API void DisposeSBTypeSummaryOptions(SBTypeSummaryOptionsRef instance);

LLDB_API bool SBTypeSummaryOptionsIsValid(SBTypeSummaryOptionsRef instance);

LLDB_API ENUM(LanguageType)
    SBTypeSummaryOptionsGetLanguage(SBTypeSummaryOptionsRef instance);

LLDB_API ENUM(TypeSummaryCapping)
    SBTypeSummaryOptionsGetCapping(SBTypeSummaryOptionsRef instance);

LLDB_API void SBTypeSummaryOptionsSetLanguage(SBTypeSummaryOptionsRef instance,
                                              ENUM(LanguageType));

LLDB_API void SBTypeSummaryOptionsSetCapping(SBTypeSummaryOptionsRef instance,
                                             ENUM(TypeSummaryCapping));

LLDB_API SBTypeSummaryRef CreateSBTypeSummary(void);

LLDB_API SBTypeSummaryRef
SBTypeSummaryCreateWithSummaryString(const char *data, uint32_t options);

LLDB_API SBTypeSummaryRef SBTypeSummaryCreateWithFunctionName(const char *data,
                                                              uint32_t options);

LLDB_API SBTypeSummaryRef SBTypeSummaryCreateWithScriptCode(const char *data,
                                                            uint32_t options);

LLDB_API SBTypeSummaryRef CloneSBTypeSummary(SBTypeSummaryRef instance);

LLDB_API void DisposeSBTypeSummary(SBTypeSummaryRef instance);

LLDB_API bool SBTypeSummaryIsValid(SBTypeSummaryRef instance);

LLDB_API bool SBTypeSummaryIsFunctionCode(SBTypeSummaryRef instance);

LLDB_API bool SBTypeSummaryIsFunctionName(SBTypeSummaryRef instance);

LLDB_API bool SBTypeSummaryIsSummaryString(SBTypeSummaryRef instance);

LLDB_API const char *SBTypeSummaryGetData(SBTypeSummaryRef instance);

LLDB_API void SBTypeSummarySetSummaryString(SBTypeSummaryRef instance,
                                            const char *data);

LLDB_API void SBTypeSummarySetFunctionName(SBTypeSummaryRef instance,
                                           const char *data);

LLDB_API void SBTypeSummarySetFunctionCode(SBTypeSummaryRef instance,
                                           const char *data);

LLDB_API uint32_t SBTypeSummaryGetOptions(SBTypeSummaryRef instance);

LLDB_API void SBTypeSummarySetOptions(SBTypeSummaryRef instance, uint32_t);

LLDB_API bool SBTypeSummaryGetDescription(SBTypeSummaryRef instance,
                                          SBStreamRef description,
                                          ENUM(DescriptionLevel)
                                              description_level);

LLDB_API bool SBTypeSummaryIsEqualTo(SBTypeSummaryRef instance,
                                     SBTypeSummaryRef rhs);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBTypeSummaryBinding_h_
