//===-- SBSourceManagerBinding.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBSourceManagerBinding_h_
#define LLDB_SBSourceManagerBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBSourceManagerRef CreateSBSourceManager(SBDebuggerRef debugger);

LLDB_API SBSourceManagerRef CreateSBSourceManager2(SBTargetRef target);

LLDB_API SBSourceManagerRef CloneSBSourceManager(SBSourceManagerRef instance);

LLDB_API void DisposeSBSourceManager(SBSourceManagerRef instance);

LLDB_API size_t SBSourceManagerDisplaySourceLinesWithLineNumbers(
    SBSourceManagerRef instance, SBFileSpecRef file, uint32_t line,
    uint32_t context_before, uint32_t context_after,
    const char *current_line_cstr, SBStreamRef s);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBSourceManagerBinding_h_
