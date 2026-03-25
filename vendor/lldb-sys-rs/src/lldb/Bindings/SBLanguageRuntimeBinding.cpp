//===-- SBLanguageRuntimeBinding.cpp ----------------------------*- C++ -*-===//
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

lldb::LanguageType
SBLanguageRuntimeGetLanguageTypeFromString(const char *string) {
  return SBLanguageRuntime::GetLanguageTypeFromString(string);
}

const char *
SBLanguageRuntimeGetNameForLanguageType(lldb::LanguageType language) {
  return SBLanguageRuntime::GetNameForLanguageType(language);
}

#ifdef __cplusplus
}
#endif
