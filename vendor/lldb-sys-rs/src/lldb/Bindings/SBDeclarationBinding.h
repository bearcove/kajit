//===-- SBDeclarationBinding.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBDeclarationBinding_h_
#define LLDB_SBDeclarationBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBDeclarationRef CreateSBDeclaration(void);

LLDB_API SBDeclarationRef CloneSBDeclaration(SBDeclarationRef instance);

LLDB_API void DisposeSBDeclaration(SBDeclarationRef instance);

LLDB_API bool SBDeclarationIsValid(SBDeclarationRef instance);

LLDB_API SBFileSpecRef SBDeclarationGetFileSpec(SBDeclarationRef instance);

LLDB_API uint32_t SBDeclarationGetLine(SBDeclarationRef instance);

LLDB_API uint32_t SBDeclarationGetColumn(SBDeclarationRef instance);

LLDB_API void SBDeclarationSetFileSpec(SBDeclarationRef instance,
                                       SBFileSpecRef filespec);

LLDB_API void SBDeclarationSetLine(SBDeclarationRef instance, uint32_t line);

LLDB_API void SBDeclarationSetColumn(SBDeclarationRef instance,
                                     uint32_t column);

LLDB_API bool SBDeclarationGetDescription(SBDeclarationRef instance,
                                          SBStreamRef description);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBDeclarationBinding_h_
