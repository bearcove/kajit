//===-- SBInstructionBinding.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBInstructionBinding_h_
#define LLDB_SBInstructionBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBInstructionRef CreateSBInstruction(void);

LLDB_API SBInstructionRef CloneSBInstruction(SBInstructionRef instance);

LLDB_API void DisposeSBInstruction(SBInstructionRef instance);

LLDB_API bool SBInstructionIsValid(SBInstructionRef instance);

LLDB_API SBAddressRef SBInstructionGetAddress(SBInstructionRef instance);

LLDB_API const char *SBInstructionGetMnemonic(SBInstructionRef instance,
                                              SBTargetRef target);

LLDB_API const char *SBInstructionGetOperands(SBInstructionRef instance,
                                              SBTargetRef target);

LLDB_API const char *SBInstructionGetComment(SBInstructionRef instance,
                                             SBTargetRef target);

LLDB_API lldb::InstructionControlFlowKind
SBInstructionGetControlFlowKind(SBInstructionRef instance, SBTargetRef target);

LLDB_API SBDataRef SBInstructionGetData(SBInstructionRef instance,
                                        SBTargetRef target);

LLDB_API size_t SBInstructionGetByteSize(SBInstructionRef instance);

LLDB_API bool SBInstructionDoesBranch(SBInstructionRef instance);

LLDB_API bool SBInstructionHasDelaySlot(SBInstructionRef instance);

LLDB_API void SBInstructionPrint(SBInstructionRef instance, SBFileRef out);

LLDB_API bool SBInstructionGetDescription(SBInstructionRef instance,
                                          SBStreamRef description);

LLDB_API bool SBInstructionEmulateWithFrame(SBInstructionRef instance,
                                            SBFrameRef frame,
                                            uint32_t evaluate_options);

LLDB_API bool SBInstructionDumpEmulation(SBInstructionRef instance,
                                         const char *triple);

LLDB_API bool SBInstructionTestEmulation(SBInstructionRef instance,
                                         SBStreamRef output_stream,
                                         const char *test_file);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBInstructionBinding_h_
