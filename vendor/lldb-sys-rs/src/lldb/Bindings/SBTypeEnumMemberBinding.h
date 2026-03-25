//===-- SBTypeEnumMemberBinding.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBTypeEnumMemberBinding_h_
#define LLDB_SBTypeEnumMemberBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBTypeEnumMemberRef CreateSBTypeEnumMember(void);

LLDB_API SBTypeEnumMemberRef
CloneSBTypeEnumMember(SBTypeEnumMemberRef instance);

LLDB_API void DisposeSBTypeEnumMember(SBTypeEnumMemberRef instance);

LLDB_API bool SBTypeEnumMemberIsValid(SBTypeEnumMemberRef instance);

LLDB_API int64_t SBTypeEnumMemberGetValueAsSigned(SBTypeEnumMemberRef instance);

LLDB_API uint64_t
SBTypeEnumMemberGetValueAsUnsigned(SBTypeEnumMemberRef instance);

LLDB_API const char *SBTypeEnumMemberGetName(SBTypeEnumMemberRef instance);

LLDB_API SBTypeRef SBTypeEnumMemberGetType(SBTypeEnumMemberRef instance);

LLDB_API bool SBTypeEnumMemberGetDescription(SBTypeEnumMemberRef instance,
                                             SBStreamRef description,
                                             ENUM(DescriptionLevel)
                                                 description_level);

LLDB_API SBTypeEnumMemberListRef CreateSBTypeEnumMemberList(void);

LLDB_API SBTypeEnumMemberListRef
CloneSBTypeEnumMemberList(SBTypeEnumMemberListRef instance);

LLDB_API void DisposeSBTypeEnumMemberList(SBTypeEnumMemberListRef instance);

LLDB_API bool SBTypeEnumMemberListIsValid(SBTypeEnumMemberListRef instance);

LLDB_API void SBTypeEnumMemberListAppend(SBTypeEnumMemberListRef instance,
                                         SBTypeEnumMemberRef entry);

LLDB_API SBTypeEnumMemberRef SBTypeEnumMemberListGetTypeEnumMemberAtIndex(
    SBTypeEnumMemberListRef instance, uint32_t index);

LLDB_API uint32_t SBTypeEnumMemberListGetSize(SBTypeEnumMemberListRef instance);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBTypeEnumMemberBinding_h_
