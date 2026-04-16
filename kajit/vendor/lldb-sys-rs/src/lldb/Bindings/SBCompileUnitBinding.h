//===-- SBCompileUnitBinding.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBCompileUnitBinding_h_
#define LLDB_SBCompileUnitBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBCompileUnitRef CreateSBCompileUnit(void);

LLDB_API SBCompileUnitRef CloneSBCompileUnit(SBCompileUnitRef instance);

LLDB_API void DisposeSBCompileUnit(SBCompileUnitRef instance);

LLDB_API bool SBCompileUnitIsValid(SBCompileUnitRef instance);

LLDB_API SBFileSpecRef SBCompileUnitGetFileSpec(SBCompileUnitRef instance);

LLDB_API uint32_t SBCompileUnitGetNumLineEntries(SBCompileUnitRef instance);

LLDB_API SBLineEntryRef
SBCompileUnitGetLineEntryAtIndex(SBCompileUnitRef instance, uint32_t idx);

LLDB_API uint32_t
SBCompileUnitFindLineEntryIndex(SBCompileUnitRef instance, uint32_t start_idx,
                                uint32_t line, SBFileSpecRef inline_file_spec);

LLDB_API uint32_t SBCompileUnitFindLineEntryIndex2(
    SBCompileUnitRef instance, uint32_t start_idx, uint32_t line,
    SBFileSpecRef inline_file_spec, bool exact);

LLDB_API SBFileSpecRef
SBCompileUnitGetSupportFileAtIndex(SBCompileUnitRef instance, uint32_t idx);

LLDB_API uint32_t SBCompileUnitGetNumSupportFiles(SBCompileUnitRef instance);

LLDB_API uint32_t SBCompileUnitFindSupportFileIndex(SBCompileUnitRef instance,
                                                    uint32_t start_idx,
                                                    SBFileSpecRef sb_file,
                                                    bool full);

LLDB_API SBTypeListRef SBCompileUnitGetTypes(SBCompileUnitRef instance,
                                             uint32_t type_mask);

LLDB_API ENUM(LanguageType) SBCompileUnitGetLanguage(SBCompileUnitRef instance);

LLDB_API bool SBCompileUnitGetDescription(SBCompileUnitRef instance,
                                          SBStreamRef description);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBCompileUnitBinding_h_
