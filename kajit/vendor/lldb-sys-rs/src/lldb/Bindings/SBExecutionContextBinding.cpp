//===-- SBExecutionContextBinding.cpp ---------------------------*- C++ -*-===//
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

SBExecutionContextRef CreateSBExecutionContext() {
  return reinterpret_cast<SBExecutionContextRef>(new SBExecutionContext());
}

SBExecutionContextRef CreateSBExecutionContext2(SBTargetRef target) {
  return reinterpret_cast<SBExecutionContextRef>(
      new SBExecutionContext(*reinterpret_cast<SBTarget *>(target)));
}

SBExecutionContextRef CreateSBExecutionContext3(SBProcessRef process) {
  return reinterpret_cast<SBExecutionContextRef>(
      new SBExecutionContext(*reinterpret_cast<SBProcess *>(process)));
}

SBExecutionContextRef CreateSBExecutionContext4(SBThreadRef thread) {
  return reinterpret_cast<SBExecutionContextRef>(
      new SBExecutionContext(*reinterpret_cast<SBThread *>(thread)));
}

SBExecutionContextRef CreateSBExecutionContext5(SBFrameRef frame) {
  return reinterpret_cast<SBExecutionContextRef>(
      new SBExecutionContext(*reinterpret_cast<SBFrame *>(frame)));
}

SBExecutionContextRef CloneSBExecutionContext(SBExecutionContextRef instance) {
  return reinterpret_cast<SBExecutionContextRef>(new SBExecutionContext(
      *reinterpret_cast<SBExecutionContext *>(instance)));
}

void DisposeSBExecutionContext(SBExecutionContextRef instance) {
  delete reinterpret_cast<SBExecutionContext *>(instance);
}

SBTargetRef SBExecutionContextGetTarget(SBExecutionContextRef instance) {
  SBExecutionContext *unwrapped =
      reinterpret_cast<SBExecutionContext *>(instance);
  return reinterpret_cast<SBTargetRef>(new SBTarget(unwrapped->GetTarget()));
}

SBProcessRef SBExecutionContextGetProcess(SBExecutionContextRef instance) {
  SBExecutionContext *unwrapped =
      reinterpret_cast<SBExecutionContext *>(instance);
  return reinterpret_cast<SBProcessRef>(new SBProcess(unwrapped->GetProcess()));
}

SBThreadRef SBExecutionContextGetThread(SBExecutionContextRef instance) {
  SBExecutionContext *unwrapped =
      reinterpret_cast<SBExecutionContext *>(instance);
  return reinterpret_cast<SBThreadRef>(new SBThread(unwrapped->GetThread()));
}

SBFrameRef SBExecutionContextGetFrame(SBExecutionContextRef instance) {
  SBExecutionContext *unwrapped =
      reinterpret_cast<SBExecutionContext *>(instance);
  return reinterpret_cast<SBFrameRef>(new SBFrame(unwrapped->GetFrame()));
}

#ifdef __cplusplus
}
#endif
