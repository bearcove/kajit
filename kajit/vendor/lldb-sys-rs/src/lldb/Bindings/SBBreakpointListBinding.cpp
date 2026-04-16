//===-- SBBreakpointListBinding.cpp -----------------------------*- C++ -*-===//
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

SBBreakpointListRef CreateSBBreakpointList(SBTargetRef target) {
  return reinterpret_cast<SBBreakpointListRef>(
      new SBBreakpointList(*reinterpret_cast<SBTarget *>(target)));
}

SBBreakpointListRef CloneSBBreakpointList(SBBreakpointListRef instance) {
  return reinterpret_cast<SBBreakpointListRef>(
      new SBBreakpointList(*reinterpret_cast<SBBreakpointList *>(instance)));
}

void DisposeSBBreakpointList(SBBreakpointListRef instance) {
  delete reinterpret_cast<SBBreakpointList *>(instance);
}

size_t SBBreakpointListGetSize(SBBreakpointListRef instance) {
  SBBreakpointList *unwrapped = reinterpret_cast<SBBreakpointList *>(instance);
  return unwrapped->GetSize();
}

SBBreakpointRef
SBBreakpointListGetBreakpointAtIndex(SBBreakpointListRef instance, size_t idx) {
  SBBreakpointList *unwrapped = reinterpret_cast<SBBreakpointList *>(instance);
  return reinterpret_cast<SBBreakpointRef>(
      new SBBreakpoint(unwrapped->GetBreakpointAtIndex(idx)));
}

SBBreakpointRef SBBreakpointListFindBreakpointByID(SBBreakpointListRef instance,
                                                   int break_id) {
  SBBreakpointList *unwrapped = reinterpret_cast<SBBreakpointList *>(instance);
  return reinterpret_cast<SBBreakpointRef>(
      new SBBreakpoint(unwrapped->FindBreakpointByID(break_id)));
}

void SBBreakpointListAppend(SBBreakpointListRef instance,
                            SBBreakpointRef sb_bkpt) {
  SBBreakpointList *unwrapped = reinterpret_cast<SBBreakpointList *>(instance);
  unwrapped->Append(*reinterpret_cast<SBBreakpoint *>(sb_bkpt));
}

bool SBBreakpointListAppendIfUnique(SBBreakpointListRef instance,
                                    SBBreakpointRef sb_bkpt) {
  SBBreakpointList *unwrapped = reinterpret_cast<SBBreakpointList *>(instance);
  return unwrapped->AppendIfUnique(*reinterpret_cast<SBBreakpoint *>(sb_bkpt));
}

void SBBreakpointListAppendByID(SBBreakpointListRef instance, int break_id) {
  SBBreakpointList *unwrapped = reinterpret_cast<SBBreakpointList *>(instance);
  unwrapped->AppendByID(break_id);
}

void SBBreakpointListClear(SBBreakpointListRef instance) {
  SBBreakpointList *unwrapped = reinterpret_cast<SBBreakpointList *>(instance);
  unwrapped->Clear();
}

#ifdef __cplusplus
}
#endif
