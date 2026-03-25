//===-- SBSourceManagerBinding.cpp ------------------------------*- C++ -*-===//
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

SBSourceManagerRef CreateSBSourceManager(SBDebuggerRef debugger) {
  return reinterpret_cast<SBSourceManagerRef>(
      new SBSourceManager(*reinterpret_cast<SBDebugger *>(debugger)));
}

SBSourceManagerRef CreateSBSourceManager2(SBTargetRef target) {
  return reinterpret_cast<SBSourceManagerRef>(
      new SBSourceManager(*reinterpret_cast<SBTarget *>(target)));
}

SBSourceManagerRef CloneSBSourceManager(SBSourceManagerRef instance) {
  return reinterpret_cast<SBSourceManagerRef>(
      new SBSourceManager(*reinterpret_cast<SBSourceManager *>(instance)));
}

void DisposeSBSourceManager(SBSourceManagerRef instance) {
  delete reinterpret_cast<SBSourceManager *>(instance);
}

size_t SBSourceManagerDisplaySourceLinesWithLineNumbers(
    SBSourceManagerRef instance, SBFileSpecRef file, uint32_t line,
    uint32_t context_before, uint32_t context_after,
    const char *current_line_cstr, SBStreamRef s) {
  SBSourceManager *unwrapped = reinterpret_cast<SBSourceManager *>(instance);
  return unwrapped->DisplaySourceLinesWithLineNumbers(
      *reinterpret_cast<SBFileSpec *>(file), line, context_before,
      context_after, current_line_cstr, *reinterpret_cast<SBStream *>(s));
}

size_t SBSourceManagerDisplaySourceLinesWithLineNumbersAndColumn(
    SBSourceManagerRef instance, SBFileSpecRef file, uint32_t line,
    uint32_t column, uint32_t context_before, uint32_t context_after,
    const char *current_line_cstr, SBStreamRef s) {
  SBSourceManager *unwrapped = reinterpret_cast<SBSourceManager *>(instance);
  return unwrapped->DisplaySourceLinesWithLineNumbersAndColumn(
      *reinterpret_cast<SBFileSpec *>(file), line, column, context_before,
      context_after, current_line_cstr, *reinterpret_cast<SBStream *>(s));
}

#ifdef __cplusplus
}
#endif
