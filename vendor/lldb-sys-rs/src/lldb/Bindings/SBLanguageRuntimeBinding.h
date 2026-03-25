//===-- SBLanguageRuntimeBinding.h ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBLanguageRuntimeBinding_h_
#define LLDB_SBLanguageRuntimeBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API lldb::LanguageType
SBLanguageRuntimeGetLanguageTypeFromString(const char *string);

LLDB_API const char *
SBLanguageRuntimeGetNameForLanguageType(lldb::LanguageType language);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBLanguageRuntimeBinding_h_
