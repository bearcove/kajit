//===-- SBValueListBinding.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBValueListBinding_h_
#define LLDB_SBValueListBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBValueListRef CreateSBValueList(void);

LLDB_API SBValueListRef CloneSBValueList(SBValueListRef instance);

LLDB_API void DisposeSBValueList(SBValueListRef instance);

LLDB_API bool SBValueListIsValid(SBValueListRef instance);

LLDB_API void SBValueListClear(SBValueListRef instance);

LLDB_API void SBValueListAppend(SBValueListRef instance, SBValueRef val_obj);

LLDB_API void SBValueListAppendList(SBValueListRef instance,
                                    SBValueListRef value_list);

LLDB_API uint32_t SBValueListGetSize(SBValueListRef instance);

LLDB_API SBValueRef SBValueListGetValueAtIndex(SBValueListRef instance,
                                               uint32_t idx);

LLDB_API SBValueRef SBValueListGetFirstValueByName(SBValueListRef instance,
                                                   const char *name);

LLDB_API SBValueRef SBValueListFindValueObjectByUID(SBValueListRef instance,
                                                    lldb_user_id_t uid);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBValueListBinding_h_
