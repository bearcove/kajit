//===-- SBValueBinding.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBValueBinding_h_
#define LLDB_SBValueBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBValueRef CreateSBValue(void);

LLDB_API SBValueRef CloneSBValue(SBValueRef instance);

LLDB_API void DisposeSBValue(SBValueRef instance);

LLDB_API bool SBValueIsValid(SBValueRef instance);

LLDB_API void SBValueClear(SBValueRef instance);

LLDB_API SBErrorRef SBValueGetError(SBValueRef instance);

LLDB_API lldb_user_id_t SBValueGetID(SBValueRef instance);

LLDB_API const char *SBValueGetName(SBValueRef instance);

LLDB_API const char *SBValueGetTypeName(SBValueRef instance);

LLDB_API const char *SBValueGetDisplayTypeName(SBValueRef instance);

LLDB_API size_t SBValueGetByteSize(SBValueRef instance);

LLDB_API bool SBValueIsInScope(SBValueRef instance);

LLDB_API ENUM(Format) SBValueGetFormat(SBValueRef instance);

LLDB_API void SBValueSetFormat(SBValueRef instance, ENUM(Format) format);

LLDB_API const char *SBValueGetValue(SBValueRef instance);

LLDB_API int64_t SBValueGetValueAsSigned(SBValueRef instance, SBErrorRef error,
                                         int64_t fail_value);

LLDB_API uint64_t SBValueGetValueAsUnsigned(SBValueRef instance,
                                            SBErrorRef error,
                                            uint64_t fail_value);

LLDB_API int64_t SBValueGetValueAsSigned2(SBValueRef instance,
                                          int64_t fail_value);

LLDB_API uint64_t SBValueGetValueAsUnsigned2(SBValueRef instance,
                                             uint64_t fail_value);

LLDB_API ENUM(ValueType) SBValueGetValueType(SBValueRef instance);

LLDB_API bool SBValueGetValueDidChange(SBValueRef instance);

LLDB_API const char *SBValueGetSummary(SBValueRef instance);

LLDB_API const char *SBValueGetSummary2(SBValueRef instance, SBStreamRef stream,
                                        SBTypeSummaryOptionsRef options);

LLDB_API const char *SBValueGetObjectDescription(SBValueRef instance);

LLDB_API SBValueRef SBValueGetDynamicValue(SBValueRef instance,
                                           ENUM(DynamicValueType) use_dynamic);

LLDB_API SBValueRef SBValueGetStaticValue(SBValueRef instance);

LLDB_API SBValueRef SBValueGetNonSyntheticValue(SBValueRef instance);

LLDB_API ENUM(DynamicValueType)
    SBValueGetPreferDynamicValue(SBValueRef instance);

LLDB_API void SBValueSetPreferDynamicValue(SBValueRef instance,
                                           ENUM(DynamicValueType) use_dynamic);

LLDB_API bool SBValueGetPreferSyntheticValue(SBValueRef instance);

LLDB_API void SBValueSetPreferSyntheticValue(SBValueRef instance,
                                             bool use_synthetic);

LLDB_API bool SBValueIsDynamic(SBValueRef instance);

LLDB_API bool SBValueIsSynthetic(SBValueRef instance);

LLDB_API bool SBValueIsSyntheticChildrenGenerated(SBValueRef instance);

LLDB_API void SBValueSetSyntheticChildrenGenerated(SBValueRef instance, bool);

LLDB_API const char *SBValueGetLocation(SBValueRef instance);

LLDB_API bool SBValueSetValueFromCString(SBValueRef instance,
                                         const char *value_str);

LLDB_API bool SBValueSetValueFromCString2(SBValueRef instance,
                                          const char *value_str,
                                          SBErrorRef error);

LLDB_API SBTypeFormatRef SBValueGetTypeFormat(SBValueRef instance);

LLDB_API SBTypeSummaryRef SBValueGetTypeSummary(SBValueRef instance);

LLDB_API SBTypeFilterRef SBValueGetTypeFilter(SBValueRef instance);

LLDB_API SBTypeSyntheticRef SBValueGetTypeSynthetic(SBValueRef instance);

LLDB_API SBValueRef SBValueGetChildAtIndex(SBValueRef instance, uint32_t idx);

LLDB_API SBValueRef SBValueCreateChildAtOffset(SBValueRef instance,
                                               const char *name,
                                               uint32_t offset, SBTypeRef type);

LLDB_API SBValueRef SBValueCast(SBValueRef instance, SBTypeRef type);

LLDB_API SBValueRef SBValueCreateValueFromExpression(SBValueRef instance,
                                                     const char *name,
                                                     const char *expression);

LLDB_API SBValueRef SBValueCreateValueFromExpression2(
    SBValueRef instance, const char *name, const char *expression,
    SBExpressionOptionsRef options);

LLDB_API SBValueRef SBValueCreateValueFromAddress(SBValueRef instance,
                                                  const char *name,
                                                  lldb_addr_t address,
                                                  SBTypeRef type);

LLDB_API SBValueRef SBValueCreateValueFromData(SBValueRef instance,
                                               const char *name, SBDataRef data,
                                               SBTypeRef type);

LLDB_API SBValueRef SBValueGetChildAtIndex2(SBValueRef instance, uint32_t idx,
                                            ENUM(DynamicValueType) use_dynamic,
                                            bool can_create_synthetic);

LLDB_API uint32_t SBValueGetIndexOfChildWithName(SBValueRef instance,
                                                 const char *name);

LLDB_API SBValueRef SBValueGetChildMemberWithName(SBValueRef instance,
                                                  const char *name);

LLDB_API SBValueRef SBValueGetChildMemberWithName2(SBValueRef instance,
                                                   const char *name,
                                                   ENUM(DynamicValueType)
                                                       use_dynamic);

LLDB_API SBValueRef SBValueGetValueForExpressionPath(SBValueRef instance,
                                                     const char *expr_path);

LLDB_API SBValueRef SBValueAddressOf(SBValueRef instance);

LLDB_API lldb_addr_t SBValueGetLoadAddress(SBValueRef instance);

LLDB_API SBAddressRef SBValueGetAddress(SBValueRef instance);

LLDB_API SBDataRef SBValueGetPointeeData(SBValueRef instance, uint32_t item_idx,
                                         uint32_t item_count);

LLDB_API SBDataRef SBValueGetData(SBValueRef instance);

LLDB_API bool SBValueSetData(SBValueRef instance, SBDataRef data,
                             SBErrorRef error);

LLDB_API SBValueRef SBValueClone(SBValueRef instance, const char *new_name);

LLDB_API SBDeclarationRef SBValueGetDeclaration(SBValueRef instance);

LLDB_API bool SBValueMightHaveChildren(SBValueRef instance);

LLDB_API bool SBValueIsRuntimeSupportValue(SBValueRef instance);

LLDB_API uint32_t SBValueGetNumChildren(SBValueRef instance);

LLDB_API void *SBValueGetOpaqueType(SBValueRef instance);

LLDB_API SBTargetRef SBValueGetTarget(SBValueRef instance);

LLDB_API SBProcessRef SBValueGetProcess(SBValueRef instance);

LLDB_API SBThreadRef SBValueGetThread(SBValueRef instance);

LLDB_API SBFrameRef SBValueGetFrame(SBValueRef instance);

LLDB_API SBValueRef SBValueDereference(SBValueRef instance);

LLDB_API bool SBValueTypeIsPointerType(SBValueRef instance);

LLDB_API SBTypeRef SBValueGetType(SBValueRef instance);

LLDB_API SBValueRef SBValuePersist(SBValueRef instance);

LLDB_API bool SBValueGetDescription(SBValueRef instance,
                                    SBStreamRef description);

LLDB_API bool SBValueGetExpressionPath(SBValueRef instance,
                                       SBStreamRef description);

LLDB_API bool SBValueGetExpressionPath2(SBValueRef instance,
                                        SBStreamRef description,
                                        bool qualify_cxx_base_classes);

LLDB_API SBValueRef SBValueEvaluateExpression(SBValueRef instance,
                                              const char *expr,
                                              SBExpressionOptionsRef options,
                                              const char *name);

LLDB_API SBWatchpointRef SBValueWatch(SBValueRef instance,
                                      bool resolve_location, bool read,
                                      bool write, SBErrorRef error);

LLDB_API SBWatchpointRef SBValueWatch2(SBValueRef instance,
                                       bool resolve_location, bool read,
                                       bool write);

LLDB_API SBWatchpointRef SBValueWatchPointee(SBValueRef instance,
                                             bool resolve_location, bool read,
                                             bool write, SBErrorRef error);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBValueBinding_h_
