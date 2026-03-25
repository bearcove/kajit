//===-- SBExecutionContextBinding.h -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBExecutionContextBinding_h_
#define LLDB_SBExecutionContextBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBExecutionContextRef CreateSBExecutionContext(void);

LLDB_API SBExecutionContextRef CreateSBExecutionContext2(SBTargetRef target);

LLDB_API SBExecutionContextRef CreateSBExecutionContext3(SBProcessRef process);

LLDB_API SBExecutionContextRef CreateSBExecutionContext4(SBThreadRef thread);

LLDB_API SBExecutionContextRef CreateSBExecutionContext5(SBFrameRef frame);

LLDB_API SBExecutionContextRef
CloneSBExecutionContext(SBExecutionContextRef instance);

LLDB_API void DisposeSBExecutionContext(SBExecutionContextRef instance);

LLDB_API SBTargetRef
SBExecutionContextGetTarget(SBExecutionContextRef instance);

LLDB_API SBProcessRef
SBExecutionContextGetProcess(SBExecutionContextRef instance);

LLDB_API SBThreadRef
SBExecutionContextGetThread(SBExecutionContextRef instance);

LLDB_API SBFrameRef SBExecutionContextGetFrame(SBExecutionContextRef instance);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBExecutionContextBinding_h_
