//===-- SBMemoryRegionInfoListBinding.h -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBMemoryRegionInfoListBinding_h_
#define LLDB_SBMemoryRegionInfoListBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBMemoryRegionInfoListRef CreateSBMemoryRegionInfoList(void);

LLDB_API SBMemoryRegionInfoListRef
CloneSBMemoryRegionInfoList(SBMemoryRegionInfoListRef instance);

LLDB_API void DisposeSBMemoryRegionInfoList(SBMemoryRegionInfoListRef instance);

LLDB_API uint32_t
SBMemoryRegionInfoListGetSize(SBMemoryRegionInfoListRef instance);

LLDB_API bool SBMemoryRegionInfoListGetMemoryRegionContainingAddress(
    SBMemoryRegionInfoListRef instance, lldb_addr_t addr,
    SBMemoryRegionInfoRef region);

LLDB_API bool
SBMemoryRegionInfoListGetMemoryRegionAtIndex(SBMemoryRegionInfoListRef instance,
                                             uint32_t idx,
                                             SBMemoryRegionInfoRef region);

LLDB_API void SBMemoryRegionInfoListAppend(SBMemoryRegionInfoListRef instance,
                                           SBMemoryRegionInfoRef region);

LLDB_API void
SBMemoryRegionInfoListAppendList(SBMemoryRegionInfoListRef instance,
                                 SBMemoryRegionInfoListRef region_list);

LLDB_API void SBMemoryRegionInfoListClear(SBMemoryRegionInfoListRef instance);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBMemoryRegionInfoListBinding_h_
