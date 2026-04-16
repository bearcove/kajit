//===-- SBValueBinding.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/LLDB.h"
#include "lldb/API/SBTypeFilter.h"
#include "lldb/API/SBTypeFormat.h"
#include "lldb/API/SBTypeSummary.h"
#include "lldb/API/SBTypeSynthetic.h"
#include "lldb/Bindings/LLDBBinding.h"

using namespace lldb;

#ifdef __cplusplus
extern "C" {
#endif

SBValueRef CreateSBValue() {
  return reinterpret_cast<SBValueRef>(new SBValue());
}

SBValueRef CloneSBValue(SBValueRef instance) {
  return reinterpret_cast<SBValueRef>(
      new SBValue(*reinterpret_cast<SBValue *>(instance)));
}

void DisposeSBValue(SBValueRef instance) {
  delete reinterpret_cast<SBValue *>(instance);
}

bool SBValueIsValid(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->IsValid();
}

void SBValueClear(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  unwrapped->Clear();
}

SBErrorRef SBValueGetError(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return reinterpret_cast<SBErrorRef>(new SBError(unwrapped->GetError()));
}

lldb_user_id_t SBValueGetID(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->GetID();
}

const char *SBValueGetName(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->GetName();
}

const char *SBValueGetTypeName(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->GetTypeName();
}

const char *SBValueGetDisplayTypeName(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->GetDisplayTypeName();
}

size_t SBValueGetByteSize(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->GetByteSize();
}

bool SBValueIsInScope(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->IsInScope();
}

enum lldb::Format SBValueGetFormat(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->GetFormat();
}

void SBValueSetFormat(SBValueRef instance, lldb::Format format) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  unwrapped->SetFormat(format);
}

const char *SBValueGetValue(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->GetValue();
}

int64_t SBValueGetValueAsSigned(SBValueRef instance, SBErrorRef error,
                                int64_t fail_value) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->GetValueAsSigned(*reinterpret_cast<SBError *>(error),
                                     fail_value);
}

uint64_t SBValueGetValueAsUnsigned(SBValueRef instance, SBErrorRef error,
                                   uint64_t fail_value) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->GetValueAsUnsigned(*reinterpret_cast<SBError *>(error),
                                       fail_value);
}

int64_t SBValueGetValueAsSigned2(SBValueRef instance, int64_t fail_value) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->GetValueAsSigned(fail_value);
}

uint64_t SBValueGetValueAsUnsigned2(SBValueRef instance, uint64_t fail_value) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->GetValueAsUnsigned(fail_value);
}

enum lldb::ValueType SBValueGetValueType(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->GetValueType();
}

bool SBValueGetValueDidChange(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->GetValueDidChange();
}

const char *SBValueGetSummary(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->GetSummary();
}

const char *SBValueGetSummary2(SBValueRef instance, SBStreamRef stream,
                               SBTypeSummaryOptionsRef options) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->GetSummary(
      *reinterpret_cast<SBStream *>(stream),
      *reinterpret_cast<SBTypeSummaryOptions *>(options));
}

const char *SBValueGetObjectDescription(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->GetObjectDescription();
}

SBValueRef SBValueGetDynamicValue(SBValueRef instance,
                                  lldb::DynamicValueType use_dynamic) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return reinterpret_cast<SBValueRef>(
      new SBValue(unwrapped->GetDynamicValue(use_dynamic)));
}

SBValueRef SBValueGetStaticValue(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return reinterpret_cast<SBValueRef>(new SBValue(unwrapped->GetStaticValue()));
}

SBValueRef SBValueGetNonSyntheticValue(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return reinterpret_cast<SBValueRef>(
      new SBValue(unwrapped->GetNonSyntheticValue()));
}

enum lldb::DynamicValueType SBValueGetPreferDynamicValue(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->GetPreferDynamicValue();
}

void SBValueSetPreferDynamicValue(SBValueRef instance,
                                  lldb::DynamicValueType use_dynamic) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  unwrapped->SetPreferDynamicValue(use_dynamic);
}

bool SBValueGetPreferSyntheticValue(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->GetPreferSyntheticValue();
}

void SBValueSetPreferSyntheticValue(SBValueRef instance, bool use_synthetic) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  unwrapped->SetPreferSyntheticValue(use_synthetic);
}

bool SBValueIsDynamic(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->IsDynamic();
}

bool SBValueIsSynthetic(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->IsSynthetic();
}

bool SBValueIsSyntheticChildrenGenerated(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->IsSyntheticChildrenGenerated();
}

void SBValueSetSyntheticChildrenGenerated(SBValueRef instance, bool b) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  unwrapped->SetSyntheticChildrenGenerated(b);
}

const char *SBValueGetLocation(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->GetLocation();
}

bool SBValueSetValueFromCString(SBValueRef instance, const char *value_str) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->SetValueFromCString(value_str);
}

bool SBValueSetValueFromCString2(SBValueRef instance, const char *value_str,
                                 SBErrorRef error) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->SetValueFromCString(value_str,
                                        *reinterpret_cast<SBError *>(error));
}

SBTypeFormatRef SBValueGetTypeFormat(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return reinterpret_cast<SBTypeFormatRef>(
      new SBTypeFormat(unwrapped->GetTypeFormat()));
}

SBTypeSummaryRef SBValueGetTypeSummary(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return reinterpret_cast<SBTypeSummaryRef>(
      new SBTypeSummary(unwrapped->GetTypeSummary()));
}

SBTypeFilterRef SBValueGetTypeFilter(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return reinterpret_cast<SBTypeFilterRef>(
      new SBTypeFilter(unwrapped->GetTypeFilter()));
}

SBTypeSyntheticRef SBValueGetTypeSynthetic(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return reinterpret_cast<SBTypeSyntheticRef>(
      new SBTypeSynthetic(unwrapped->GetTypeSynthetic()));
}

SBValueRef SBValueGetChildAtIndex(SBValueRef instance, uint32_t idx) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return reinterpret_cast<SBValueRef>(
      new SBValue(unwrapped->GetChildAtIndex(idx)));
}

SBValueRef SBValueCreateChildAtOffset(SBValueRef instance, const char *name,
                                      uint32_t offset, SBTypeRef type) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return reinterpret_cast<SBValueRef>(
      new SBValue(unwrapped->CreateChildAtOffset(
          name, offset, *reinterpret_cast<SBType *>(type))));
}

SBValueRef SBValueCast(SBValueRef instance, SBTypeRef type) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return reinterpret_cast<SBValueRef>(
      new SBValue(unwrapped->Cast(*reinterpret_cast<SBType *>(type))));
}

SBValueRef SBValueCreateValueFromExpression(SBValueRef instance,
                                            const char *name,
                                            const char *expression) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return reinterpret_cast<SBValueRef>(
      new SBValue(unwrapped->CreateValueFromExpression(name, expression)));
}

SBValueRef SBValueCreateValueFromExpression2(SBValueRef instance,
                                             const char *name,
                                             const char *expression,
                                             SBExpressionOptionsRef options) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return reinterpret_cast<SBValueRef>(
      new SBValue(unwrapped->CreateValueFromExpression(
          name, expression,
          *reinterpret_cast<SBExpressionOptions *>(options))));
}

SBValueRef SBValueCreateValueFromAddress(SBValueRef instance, const char *name,
                                         lldb_addr_t address, SBTypeRef type) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return reinterpret_cast<SBValueRef>(
      new SBValue(unwrapped->CreateValueFromAddress(
          name, address, *reinterpret_cast<SBType *>(type))));
}

SBValueRef SBValueCreateValueFromData(SBValueRef instance, const char *name,
                                      SBDataRef data, SBTypeRef type) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return reinterpret_cast<SBValueRef>(new SBValue(
      unwrapped->CreateValueFromData(name, *reinterpret_cast<SBData *>(data),
                                     *reinterpret_cast<SBType *>(type))));
}

SBValueRef SBValueGetChildAtIndex2(SBValueRef instance, uint32_t idx,
                                   lldb::DynamicValueType use_dynamic,
                                   bool can_create_synthetic) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return reinterpret_cast<SBValueRef>(new SBValue(
      unwrapped->GetChildAtIndex(idx, use_dynamic, can_create_synthetic)));
}

uint32_t SBValueGetIndexOfChildWithName(SBValueRef instance, const char *name) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->GetIndexOfChildWithName(name);
}

SBValueRef SBValueGetChildMemberWithName(SBValueRef instance,
                                         const char *name) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return reinterpret_cast<SBValueRef>(
      new SBValue(unwrapped->GetChildMemberWithName(name)));
}

SBValueRef SBValueGetChildMemberWithName2(SBValueRef instance, const char *name,
                                          lldb::DynamicValueType use_dynamic) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return reinterpret_cast<SBValueRef>(
      new SBValue(unwrapped->GetChildMemberWithName(name, use_dynamic)));
}

SBValueRef SBValueGetValueForExpressionPath(SBValueRef instance,
                                            const char *expr_path) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return reinterpret_cast<SBValueRef>(
      new SBValue(unwrapped->GetValueForExpressionPath(expr_path)));
}

SBValueRef SBValueAddressOf(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return reinterpret_cast<SBValueRef>(new SBValue(unwrapped->AddressOf()));
}

lldb_addr_t SBValueGetLoadAddress(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->GetLoadAddress();
}

SBAddressRef SBValueGetAddress(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return reinterpret_cast<SBAddressRef>(new SBAddress(unwrapped->GetAddress()));
}

SBDataRef SBValueGetPointeeData(SBValueRef instance, uint32_t item_idx,
                                uint32_t item_count) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return reinterpret_cast<SBDataRef>(
      new SBData(unwrapped->GetPointeeData(item_idx, item_count)));
}

SBDataRef SBValueGetData(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return reinterpret_cast<SBDataRef>(new SBData(unwrapped->GetData()));
}

bool SBValueSetData(SBValueRef instance, SBDataRef data, SBErrorRef error) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->SetData(*reinterpret_cast<SBData *>(data),
                            *reinterpret_cast<SBError *>(error));
}

SBValueRef SBValueClone(SBValueRef instance, const char *new_name) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return reinterpret_cast<SBValueRef>(new SBValue(unwrapped->Clone(new_name)));
}

SBDeclarationRef SBValueGetDeclaration(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return reinterpret_cast<SBDeclarationRef>(
      new SBDeclaration(unwrapped->GetDeclaration()));
}

bool SBValueMightHaveChildren(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->MightHaveChildren();
}

bool SBValueIsRuntimeSupportValue(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->IsRuntimeSupportValue();
}

uint32_t SBValueGetNumChildren(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->GetNumChildren();
}

void *SBValueGetOpaqueType(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->GetOpaqueType();
}

SBTargetRef SBValueGetTarget(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return reinterpret_cast<SBTargetRef>(new SBTarget(unwrapped->GetTarget()));
}

SBProcessRef SBValueGetProcess(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return reinterpret_cast<SBProcessRef>(new SBProcess(unwrapped->GetProcess()));
}

SBThreadRef SBValueGetThread(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return reinterpret_cast<SBThreadRef>(new SBThread(unwrapped->GetThread()));
}

SBFrameRef SBValueGetFrame(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return reinterpret_cast<SBFrameRef>(new SBFrame(unwrapped->GetFrame()));
}

SBValueRef SBValueDereference(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return reinterpret_cast<SBValueRef>(new SBValue(unwrapped->Dereference()));
}

bool SBValueTypeIsPointerType(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->TypeIsPointerType();
}

SBTypeRef SBValueGetType(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return reinterpret_cast<SBTypeRef>(new SBType(unwrapped->GetType()));
}

SBValueRef SBValuePersist(SBValueRef instance) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return reinterpret_cast<SBValueRef>(new SBValue(unwrapped->Persist()));
}

bool SBValueGetDescription(SBValueRef instance, SBStreamRef description) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description));
}

bool SBValueGetExpressionPath(SBValueRef instance, SBStreamRef description) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->GetExpressionPath(
      *reinterpret_cast<SBStream *>(description));
}

bool SBValueGetExpressionPath2(SBValueRef instance, SBStreamRef description,
                               bool qualify_cxx_base_classes) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return unwrapped->GetExpressionPath(
      *reinterpret_cast<SBStream *>(description), qualify_cxx_base_classes);
}

SBValueRef SBValueEvaluateExpression(SBValueRef instance, const char *expr,
                                     SBExpressionOptionsRef options,
                                     const char *name) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return reinterpret_cast<SBValueRef>(new SBValue(unwrapped->EvaluateExpression(
      expr, *reinterpret_cast<SBExpressionOptions *>(options), name)));
}

SBWatchpointRef SBValueWatch(SBValueRef instance, bool resolve_location,
                             bool read, bool write, SBErrorRef error) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return reinterpret_cast<SBWatchpointRef>(new SBWatchpoint(unwrapped->Watch(
      resolve_location, read, write, *reinterpret_cast<SBError *>(error))));
}

SBWatchpointRef SBValueWatch2(SBValueRef instance, bool resolve_location,
                              bool read, bool write) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return reinterpret_cast<SBWatchpointRef>(
      new SBWatchpoint(unwrapped->Watch(resolve_location, read, write)));
}

SBWatchpointRef SBValueWatchPointee(SBValueRef instance, bool resolve_location,
                                    bool read, bool write, SBErrorRef error) {
  SBValue *unwrapped = reinterpret_cast<SBValue *>(instance);
  return reinterpret_cast<SBWatchpointRef>(
      new SBWatchpoint(unwrapped->WatchPointee(
          resolve_location, read, write, *reinterpret_cast<SBError *>(error))));
}

#ifdef __cplusplus
}
#endif
