//===-- SBTypeEnumMemberBinding.cpp -----------------------------*- C++ -*-===//
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

SBTypeEnumMemberRef CreateSBTypeEnumMember() {
  return reinterpret_cast<SBTypeEnumMemberRef>(new SBTypeEnumMember());
}

SBTypeEnumMemberRef CloneSBTypeEnumMember(SBTypeEnumMemberRef instance) {
  return reinterpret_cast<SBTypeEnumMemberRef>(
      new SBTypeEnumMember(*reinterpret_cast<SBTypeEnumMember *>(instance)));
}

void DisposeSBTypeEnumMember(SBTypeEnumMemberRef instance) {
  delete reinterpret_cast<SBTypeEnumMember *>(instance);
}

bool SBTypeEnumMemberIsValid(SBTypeEnumMemberRef instance) {
  SBTypeEnumMember *unwrapped = reinterpret_cast<SBTypeEnumMember *>(instance);
  return unwrapped->IsValid();
}

int64_t SBTypeEnumMemberGetValueAsSigned(SBTypeEnumMemberRef instance) {
  SBTypeEnumMember *unwrapped = reinterpret_cast<SBTypeEnumMember *>(instance);
  return unwrapped->GetValueAsSigned();
}

uint64_t SBTypeEnumMemberGetValueAsUnsigned(SBTypeEnumMemberRef instance) {
  SBTypeEnumMember *unwrapped = reinterpret_cast<SBTypeEnumMember *>(instance);
  return unwrapped->GetValueAsUnsigned();
}

const char *SBTypeEnumMemberGetName(SBTypeEnumMemberRef instance) {
  SBTypeEnumMember *unwrapped = reinterpret_cast<SBTypeEnumMember *>(instance);
  return unwrapped->GetName();
}

SBTypeRef SBTypeEnumMemberGetType(SBTypeEnumMemberRef instance) {
  SBTypeEnumMember *unwrapped = reinterpret_cast<SBTypeEnumMember *>(instance);
  return reinterpret_cast<SBTypeRef>(new SBType(unwrapped->GetType()));
}

bool SBTypeEnumMemberGetDescription(SBTypeEnumMemberRef instance,
                                    SBStreamRef description,
                                    lldb::DescriptionLevel description_level) {
  SBTypeEnumMember *unwrapped = reinterpret_cast<SBTypeEnumMember *>(instance);
  return unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description),
                                   description_level);
}

SBTypeEnumMemberListRef CreateSBTypeEnumMemberList() {
  return reinterpret_cast<SBTypeEnumMemberListRef>(new SBTypeEnumMemberList());
}

SBTypeEnumMemberListRef
CloneSBTypeEnumMemberList(SBTypeEnumMemberListRef instance) {
  return reinterpret_cast<SBTypeEnumMemberListRef>(new SBTypeEnumMemberList(
      *reinterpret_cast<SBTypeEnumMemberList *>(instance)));
}

void DisposeSBTypeEnumMemberList(SBTypeEnumMemberListRef instance) {
  delete reinterpret_cast<SBTypeEnumMemberList *>(instance);
}

bool SBTypeEnumMemberListIsValid(SBTypeEnumMemberListRef instance) {
  SBTypeEnumMemberList *unwrapped =
      reinterpret_cast<SBTypeEnumMemberList *>(instance);
  return unwrapped->IsValid();
}

void SBTypeEnumMemberListAppend(SBTypeEnumMemberListRef instance,
                                SBTypeEnumMemberRef entry) {
  SBTypeEnumMemberList *unwrapped =
      reinterpret_cast<SBTypeEnumMemberList *>(instance);
  unwrapped->Append(*reinterpret_cast<SBTypeEnumMember *>(entry));
}

SBTypeEnumMemberRef
SBTypeEnumMemberListGetTypeEnumMemberAtIndex(SBTypeEnumMemberListRef instance,
                                             uint32_t index) {
  SBTypeEnumMemberList *unwrapped =
      reinterpret_cast<SBTypeEnumMemberList *>(instance);
  return reinterpret_cast<SBTypeEnumMemberRef>(
      new SBTypeEnumMember(unwrapped->GetTypeEnumMemberAtIndex(index)));
}

uint32_t SBTypeEnumMemberListGetSize(SBTypeEnumMemberListRef instance) {
  SBTypeEnumMemberList *unwrapped =
      reinterpret_cast<SBTypeEnumMemberList *>(instance);
  return unwrapped->GetSize();
}

#ifdef __cplusplus
}
#endif
