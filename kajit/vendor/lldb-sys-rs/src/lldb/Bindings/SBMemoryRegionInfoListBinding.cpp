//===-- SBMemoryRegionInfoListBinding.cpp -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/LLDB.h"
#include "lldb/Bindings/LLDBBinding.h"

using namespace lldb;

#ifdef __cplusplus
extern "C" {
#endif

SBMemoryRegionInfoListRef CreateSBMemoryRegionInfoList() {
  return reinterpret_cast<SBMemoryRegionInfoListRef>(
      new SBMemoryRegionInfoList());
}

SBMemoryRegionInfoListRef
CloneSBMemoryRegionInfoList(SBMemoryRegionInfoListRef instance) {
  return reinterpret_cast<SBMemoryRegionInfoListRef>(new SBMemoryRegionInfoList(
      *reinterpret_cast<SBMemoryRegionInfoList *>(instance)));
}

void DisposeSBMemoryRegionInfoList(SBMemoryRegionInfoListRef instance) {
  delete reinterpret_cast<SBMemoryRegionInfoList *>(instance);
}

uint32_t SBMemoryRegionInfoListGetSize(SBMemoryRegionInfoListRef instance) {
  SBMemoryRegionInfoList *unwrapped =
      reinterpret_cast<SBMemoryRegionInfoList *>(instance);
  return unwrapped->GetSize();
}

bool SBMemoryRegionInfoListGetMemoryRegionContainingAddress(
    SBMemoryRegionInfoListRef instance, lldb_addr_t addr,
    SBMemoryRegionInfoRef region) {
  SBMemoryRegionInfoList *unwrapped =
      reinterpret_cast<SBMemoryRegionInfoList *>(instance);
  return unwrapped->GetMemoryRegionContainingAddress(
      addr, *reinterpret_cast<SBMemoryRegionInfo *>(region));
}

bool SBMemoryRegionInfoListGetMemoryRegionAtIndex(
    SBMemoryRegionInfoListRef instance, uint32_t idx,
    SBMemoryRegionInfoRef region) {
  SBMemoryRegionInfoList *unwrapped =
      reinterpret_cast<SBMemoryRegionInfoList *>(instance);
  return unwrapped->GetMemoryRegionAtIndex(
      idx, *reinterpret_cast<SBMemoryRegionInfo *>(region));
}

void SBMemoryRegionInfoListAppend(SBMemoryRegionInfoListRef instance,
                                  SBMemoryRegionInfoRef region) {
  SBMemoryRegionInfoList *unwrapped =
      reinterpret_cast<SBMemoryRegionInfoList *>(instance);
  unwrapped->Append(*reinterpret_cast<SBMemoryRegionInfo *>(region));
}

void SBMemoryRegionInfoListAppendList(SBMemoryRegionInfoListRef instance,
                                      SBMemoryRegionInfoListRef region_list) {
  SBMemoryRegionInfoList *unwrapped =
      reinterpret_cast<SBMemoryRegionInfoList *>(instance);
  unwrapped->Append(*reinterpret_cast<SBMemoryRegionInfoList *>(region_list));
}

void SBMemoryRegionInfoListClear(SBMemoryRegionInfoListRef instance) {
  SBMemoryRegionInfoList *unwrapped =
      reinterpret_cast<SBMemoryRegionInfoList *>(instance);
  unwrapped->Clear();
}

#ifdef __cplusplus
}
#endif
