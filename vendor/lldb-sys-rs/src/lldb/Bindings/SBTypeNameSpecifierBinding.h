//===-- SBTypeNameSpecifierBinding.h ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBTypeNameSpecifierBinding_h_
#define LLDB_SBTypeNameSpecifierBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBTypeNameSpecifierRef CreateSBTypeNameSpecifier(void);

LLDB_API SBTypeNameSpecifierRef CreateSBTypeNameSpecifier2(const char *name,
                                                           bool is_regex);

LLDB_API SBTypeNameSpecifierRef CreateSBTypeNameSpecifier3(SBTypeRef type);

LLDB_API SBTypeNameSpecifierRef
CloneSBTypeNameSpecifier(SBTypeNameSpecifierRef instance);

LLDB_API void DisposeSBTypeNameSpecifier(SBTypeNameSpecifierRef instance);

LLDB_API bool SBTypeNameSpecifierIsValid(SBTypeNameSpecifierRef instance);

LLDB_API const char *
SBTypeNameSpecifierGetName(SBTypeNameSpecifierRef instance);

LLDB_API SBTypeRef SBTypeNameSpecifierGetType(SBTypeNameSpecifierRef instance);

LLDB_API bool SBTypeNameSpecifierIsRegex(SBTypeNameSpecifierRef instance);

LLDB_API bool SBTypeNameSpecifierGetDescription(SBTypeNameSpecifierRef instance,
                                                SBStreamRef description,
                                                ENUM(DescriptionLevel)
                                                    description_level);

LLDB_API bool SBTypeNameSpecifierIsEqualTo(SBTypeNameSpecifierRef instance,
                                           SBTypeNameSpecifierRef rhs);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBTypeNameSpecifierBinding_h_
