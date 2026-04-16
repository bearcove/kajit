//===-- SBTypeBinding.cpp ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/LLDB.h"
#include "lldb/API/SBTypeEnumMember.h"
#include "lldb/Bindings/LLDBBinding.h"

using namespace lldb;

#ifdef __cplusplus
extern "C" {
#endif

SBTypeMemberRef CreateSBTypeMember() {
  return reinterpret_cast<SBTypeMemberRef>(new SBTypeMember());
}

SBTypeMemberRef CloneSBTypeMember(SBTypeMemberRef instance) {
  return reinterpret_cast<SBTypeMemberRef>(
      new SBTypeMember(*reinterpret_cast<SBTypeMember *>(instance)));
}

void DisposeSBTypeMember(SBTypeMemberRef instance) {
  delete reinterpret_cast<SBTypeMember *>(instance);
}

bool SBTypeMemberIsValid(SBTypeMemberRef instance) {
  SBTypeMember *unwrapped = reinterpret_cast<SBTypeMember *>(instance);
  return unwrapped->IsValid();
}

const char *SBTypeMemberGetName(SBTypeMemberRef instance) {
  SBTypeMember *unwrapped = reinterpret_cast<SBTypeMember *>(instance);
  return unwrapped->GetName();
}

SBTypeRef SBTypeMemberGetType(SBTypeMemberRef instance) {
  SBTypeMember *unwrapped = reinterpret_cast<SBTypeMember *>(instance);
  return reinterpret_cast<SBTypeRef>(new SBType(unwrapped->GetType()));
}

uint64_t SBTypeMemberGetOffsetInBytes(SBTypeMemberRef instance) {
  SBTypeMember *unwrapped = reinterpret_cast<SBTypeMember *>(instance);
  return unwrapped->GetOffsetInBytes();
}

uint64_t SBTypeMemberGetOffsetInBits(SBTypeMemberRef instance) {
  SBTypeMember *unwrapped = reinterpret_cast<SBTypeMember *>(instance);
  return unwrapped->GetOffsetInBits();
}

bool SBTypeMemberIsBitfield(SBTypeMemberRef instance) {
  SBTypeMember *unwrapped = reinterpret_cast<SBTypeMember *>(instance);
  return unwrapped->IsBitfield();
}

uint32_t SBTypeMemberGetBitfieldSizeInBits(SBTypeMemberRef instance) {
  SBTypeMember *unwrapped = reinterpret_cast<SBTypeMember *>(instance);
  return unwrapped->GetBitfieldSizeInBits();
}

bool SBTypeMemberGetDescription(SBTypeMemberRef instance,
                                SBStreamRef description,
                                lldb::DescriptionLevel description_level) {
  SBTypeMember *unwrapped = reinterpret_cast<SBTypeMember *>(instance);
  return unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description),
                                   description_level);
}

SBTypeMemberFunctionRef CreateSBTypeMemberFunction() {
  return reinterpret_cast<SBTypeMemberFunctionRef>(new SBTypeMemberFunction());
}

SBTypeMemberFunctionRef
CloneSBTypeMemberFunction(SBTypeMemberFunctionRef instance) {
  return reinterpret_cast<SBTypeMemberFunctionRef>(new SBTypeMemberFunction(
      *reinterpret_cast<SBTypeMemberFunction *>(instance)));
}

void DisposeSBTypeMemberFunction(SBTypeMemberFunctionRef instance) {
  delete reinterpret_cast<SBTypeMemberFunction *>(instance);
}

bool SBTypeMemberFunctionIsValid(SBTypeMemberFunctionRef instance) {
  SBTypeMemberFunction *unwrapped =
      reinterpret_cast<SBTypeMemberFunction *>(instance);
  return unwrapped->IsValid();
}

const char *SBTypeMemberFunctionGetName(SBTypeMemberFunctionRef instance) {
  SBTypeMemberFunction *unwrapped =
      reinterpret_cast<SBTypeMemberFunction *>(instance);
  return unwrapped->GetName();
}

SBTypeRef SBTypeMemberFunctionGetType(SBTypeMemberFunctionRef instance) {
  SBTypeMemberFunction *unwrapped =
      reinterpret_cast<SBTypeMemberFunction *>(instance);
  return reinterpret_cast<SBTypeRef>(new SBType(unwrapped->GetType()));
}

SBTypeRef SBTypeMemberFunctionGetReturnType(SBTypeMemberFunctionRef instance) {
  SBTypeMemberFunction *unwrapped =
      reinterpret_cast<SBTypeMemberFunction *>(instance);
  return reinterpret_cast<SBTypeRef>(new SBType(unwrapped->GetReturnType()));
}

uint32_t
SBTypeMemberFunctionGetNumberOfArguments(SBTypeMemberFunctionRef instance) {
  SBTypeMemberFunction *unwrapped =
      reinterpret_cast<SBTypeMemberFunction *>(instance);
  return unwrapped->GetNumberOfArguments();
}

SBTypeRef
SBTypeMemberFunctionGetArgumentTypeAtIndex(SBTypeMemberFunctionRef instance,
                                           uint32_t var0) {
  SBTypeMemberFunction *unwrapped =
      reinterpret_cast<SBTypeMemberFunction *>(instance);
  return reinterpret_cast<SBTypeRef>(
      new SBType(unwrapped->GetArgumentTypeAtIndex(var0)));
}

enum lldb::MemberFunctionKind
SBTypeMemberFunctionGetKind(SBTypeMemberFunctionRef instance) {
  SBTypeMemberFunction *unwrapped =
      reinterpret_cast<SBTypeMemberFunction *>(instance);
  return unwrapped->GetKind();
}

bool SBTypeMemberFunctionGetDescription(
    SBTypeMemberFunctionRef instance, SBStreamRef description,
    lldb::DescriptionLevel description_level) {
  SBTypeMemberFunction *unwrapped =
      reinterpret_cast<SBTypeMemberFunction *>(instance);
  return unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description),
                                   description_level);
}

SBTypeRef CreateSBType() { return reinterpret_cast<SBTypeRef>(new SBType()); }

SBTypeRef CloneSBType(SBTypeRef instance) {
  return reinterpret_cast<SBTypeRef>(
      new SBType(*reinterpret_cast<SBType *>(instance)));
}

void DisposeSBType(SBTypeRef instance) {
  delete reinterpret_cast<SBType *>(instance);
}

bool SBTypeIsValid(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return unwrapped->IsValid();
}

uint64_t SBTypeGetByteSize(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return unwrapped->GetByteSize();
}

bool SBTypeIsPointerType(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return unwrapped->IsPointerType();
}

bool SBTypeIsReferenceType(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return unwrapped->IsReferenceType();
}

bool SBTypeIsFunctionType(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return unwrapped->IsFunctionType();
}

bool SBTypeIsPolymorphicClass(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return unwrapped->IsPolymorphicClass();
}

bool SBTypeIsArrayType(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return unwrapped->IsArrayType();
}

bool SBTypeIsVectorType(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return unwrapped->IsVectorType();
}

bool SBTypeIsTypedefType(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return unwrapped->IsTypedefType();
}

bool SBTypeIsAnonymousType(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return unwrapped->IsAnonymousType();
}

bool SBTypeIsScopedEnumerationType(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return unwrapped->IsScopedEnumerationType();
}

bool SBTypeIsAggregateType(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return unwrapped->IsAggregateType();
}

SBTypeRef SBTypeGetPointerType(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return reinterpret_cast<SBTypeRef>(new SBType(unwrapped->GetPointerType()));
}

SBTypeRef SBTypeGetPointeeType(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return reinterpret_cast<SBTypeRef>(new SBType(unwrapped->GetPointeeType()));
}

SBTypeRef SBTypeGetReferenceType(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return reinterpret_cast<SBTypeRef>(new SBType(unwrapped->GetReferenceType()));
}

SBTypeRef SBTypeGetTypedefedType(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return reinterpret_cast<SBTypeRef>(new SBType(unwrapped->GetTypedefedType()));
}

SBTypeRef SBTypeGetDereferencedType(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return reinterpret_cast<SBTypeRef>(
      new SBType(unwrapped->GetDereferencedType()));
}

SBTypeRef SBTypeGetUnqualifiedType(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return reinterpret_cast<SBTypeRef>(
      new SBType(unwrapped->GetUnqualifiedType()));
}

SBTypeRef SBTypeGetArrayElementType(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return reinterpret_cast<SBTypeRef>(
      new SBType(unwrapped->GetArrayElementType()));
}

SBTypeRef SBTypeGetArrayType(SBTypeRef instance, uint64_t size) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return reinterpret_cast<SBTypeRef>(new SBType(unwrapped->GetArrayType(size)));
}

SBTypeRef SBTypeGetVectorElementType(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return reinterpret_cast<SBTypeRef>(
      new SBType(unwrapped->GetVectorElementType()));
}

SBTypeRef SBTypeGetCanonicalType(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return reinterpret_cast<SBTypeRef>(new SBType(unwrapped->GetCanonicalType()));
}

SBTypeRef SBTypeGetEnumerationIntegerType(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return reinterpret_cast<SBTypeRef>(
      new SBType(unwrapped->GetEnumerationIntegerType()));
}

enum lldb::BasicType SBTypeGetBasicType(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return unwrapped->GetBasicType();
}

SBTypeRef SBTypeGetBasicType2(SBTypeRef instance, lldb::BasicType type) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return reinterpret_cast<SBTypeRef>(new SBType(unwrapped->GetBasicType(type)));
}

uint32_t SBTypeGetNumberOfFields(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return unwrapped->GetNumberOfFields();
}

uint32_t SBTypeGetNumberOfDirectBaseClasses(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return unwrapped->GetNumberOfDirectBaseClasses();
}

uint32_t SBTypeGetNumberOfVirtualBaseClasses(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return unwrapped->GetNumberOfVirtualBaseClasses();
}

SBTypeMemberRef SBTypeGetFieldAtIndex(SBTypeRef instance, uint32_t idx) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return reinterpret_cast<SBTypeMemberRef>(
      new SBTypeMember(unwrapped->GetFieldAtIndex(idx)));
}

SBTypeMemberRef SBTypeGetDirectBaseClassAtIndex(SBTypeRef instance,
                                                uint32_t idx) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return reinterpret_cast<SBTypeMemberRef>(
      new SBTypeMember(unwrapped->GetDirectBaseClassAtIndex(idx)));
}

SBTypeMemberRef SBTypeGetVirtualBaseClassAtIndex(SBTypeRef instance,
                                                 uint32_t idx) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return reinterpret_cast<SBTypeMemberRef>(
      new SBTypeMember(unwrapped->GetVirtualBaseClassAtIndex(idx)));
}

SBTypeEnumMemberListRef SBTypeGetEnumMembers(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return reinterpret_cast<SBTypeEnumMemberListRef>(
      new SBTypeEnumMemberList(unwrapped->GetEnumMembers()));
}

uint32_t SBTypeGetNumberOfTemplateArguments(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return unwrapped->GetNumberOfTemplateArguments();
}

SBTypeRef SBTypeGetTemplateArgumentType(SBTypeRef instance, uint32_t idx) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return reinterpret_cast<SBTypeRef>(
      new SBType(unwrapped->GetTemplateArgumentType(idx)));
}

lldb::TemplateArgumentKind SBTypeGetTemplateArgumentKind(SBTypeRef instance,
                                                         uint32_t idx) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return unwrapped->GetTemplateArgumentKind(idx);
}

SBTypeRef SBTypeGetFunctionReturnType(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return reinterpret_cast<SBTypeRef>(
      new SBType(unwrapped->GetFunctionReturnType()));
}

SBTypeListRef SBTypeGetFunctionArgumentTypes(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return reinterpret_cast<SBTypeListRef>(
      new SBTypeList(unwrapped->GetFunctionArgumentTypes()));
}

uint32_t SBTypeGetNumberOfMemberFunctions(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return unwrapped->GetNumberOfMemberFunctions();
}

SBTypeMemberFunctionRef SBTypeGetMemberFunctionAtIndex(SBTypeRef instance,
                                                       uint32_t idx) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return reinterpret_cast<SBTypeMemberFunctionRef>(
      new SBTypeMemberFunction(unwrapped->GetMemberFunctionAtIndex(idx)));
}

SBModuleRef SBTypeGetModule(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return reinterpret_cast<SBModuleRef>(new SBModule(unwrapped->GetModule()));
}

const char *SBTypeGetName(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return unwrapped->GetName();
}

const char *SBTypeGetDisplayTypeName(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return unwrapped->GetDisplayTypeName();
}

lldb::TypeClass SBTypeGetTypeClass(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return unwrapped->GetTypeClass();
}

bool SBTypeIsTypeComplete(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return unwrapped->IsTypeComplete();
}

uint32_t SBTypeGetTypeFlags(SBTypeRef instance) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return unwrapped->GetTypeFlags();
}

bool SBTypeGetDescription(SBTypeRef instance, SBStreamRef description,
                          lldb::DescriptionLevel description_level) {
  SBType *unwrapped = reinterpret_cast<SBType *>(instance);
  return unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description),
                                   description_level);
}

SBTypeListRef CreateSBTypeList() {
  return reinterpret_cast<SBTypeListRef>(new SBTypeList());
}

SBTypeListRef CloneSBTypeList(SBTypeListRef instance) {
  return reinterpret_cast<SBTypeListRef>(
      new SBTypeList(*reinterpret_cast<SBTypeList *>(instance)));
}

void DisposeSBTypeList(SBTypeListRef instance) {
  delete reinterpret_cast<SBTypeList *>(instance);
}

bool SBTypeListIsValid(SBTypeListRef instance) {
  SBTypeList *unwrapped = reinterpret_cast<SBTypeList *>(instance);
  return unwrapped->IsValid();
}

void SBTypeListAppend(SBTypeListRef instance, SBTypeRef type) {
  SBTypeList *unwrapped = reinterpret_cast<SBTypeList *>(instance);
  unwrapped->Append(*reinterpret_cast<SBType *>(type));
}

SBTypeRef SBTypeListGetTypeAtIndex(SBTypeListRef instance, uint32_t index) {
  SBTypeList *unwrapped = reinterpret_cast<SBTypeList *>(instance);
  return reinterpret_cast<SBTypeRef>(
      new SBType(unwrapped->GetTypeAtIndex(index)));
}

uint32_t SBTypeListGetSize(SBTypeListRef instance) {
  SBTypeList *unwrapped = reinterpret_cast<SBTypeList *>(instance);
  return unwrapped->GetSize();
}

#ifdef __cplusplus
}
#endif
