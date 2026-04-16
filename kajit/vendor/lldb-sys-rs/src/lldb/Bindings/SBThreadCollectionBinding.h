//===-- SBThreadCollectionBinding.h -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBThreadCollectionBinding_h_
#define LLDB_SBThreadCollectionBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBThreadCollectionRef CreateSBThreadCollection(void);

LLDB_API SBThreadCollectionRef
CloneSBThreadCollection(SBThreadCollectionRef instance);

LLDB_API void DisposeSBThreadCollection(SBThreadCollectionRef instance);

LLDB_API bool SBThreadCollectionIsValid(SBThreadCollectionRef instance);

LLDB_API size_t SBThreadCollectionGetSize(SBThreadCollectionRef instance);

LLDB_API SBThreadRef
SBThreadCollectionGetThreadAtIndex(SBThreadCollectionRef instance, size_t idx);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBThreadCollectionBinding_h_
