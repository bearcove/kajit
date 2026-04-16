//===-- SBInstructionListBinding.h ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBInstructionListBinding_h_
#define LLDB_SBInstructionListBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBInstructionListRef CreateSBInstructionList(void);

LLDB_API SBInstructionListRef
CloneSBInstructionList(SBInstructionListRef instance);

LLDB_API void DisposeSBInstructionList(SBInstructionListRef instance);

LLDB_API bool SBInstructionListIsValid(SBInstructionListRef instance);

LLDB_API size_t SBInstructionListGetSize(SBInstructionListRef instance);

LLDB_API SBInstructionRef SBInstructionListGetInstructionAtIndex(
    SBInstructionListRef instance, uint32_t idx);

LLDB_API void SBInstructionListClear(SBInstructionListRef instance);

LLDB_API void SBInstructionListAppendInstruction(SBInstructionListRef instance,
                                                 SBInstructionRef inst);

LLDB_API void SBInstructionListPrint(SBInstructionListRef instance,
                                     SBFileRef out);

LLDB_API bool SBInstructionListGetDescription(SBInstructionListRef instance,
                                              SBStreamRef description);

LLDB_API bool
SBInstructionListDumpEmulationForAllInstructions(SBInstructionListRef instance,
                                                 const char *triple);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBInstructionListBinding_h_
