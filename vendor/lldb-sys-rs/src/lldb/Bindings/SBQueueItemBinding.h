//===-- SBQueueItemBinding.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBQueueItemBinding_h_
#define LLDB_SBQueueItemBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBQueueItemRef CreateSBQueueItem(void);

LLDB_API SBQueueItemRef CloneSBQueueItem(SBQueueItemRef instance);

LLDB_API void DisposeSBQueueItem(SBQueueItemRef instance);

LLDB_API bool SBQueueItemIsValid(SBQueueItemRef instance);

LLDB_API void SBQueueItemClear(SBQueueItemRef instance);

LLDB_API ENUM(QueueItemKind) SBQueueItemGetKind(SBQueueItemRef instance);

LLDB_API void SBQueueItemSetKind(SBQueueItemRef instance,
                                 ENUM(QueueItemKind) kind);

LLDB_API SBAddressRef SBQueueItemGetAddress(SBQueueItemRef instance);

LLDB_API void SBQueueItemSetAddress(SBQueueItemRef instance, SBAddressRef addr);

LLDB_API SBThreadRef SBQueueItemGetExtendedBacktraceThread(
    SBQueueItemRef instance, const char *type);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBQueueItemBinding_h_
