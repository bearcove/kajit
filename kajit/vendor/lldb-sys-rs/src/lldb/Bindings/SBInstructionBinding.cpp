//===-- SBInstructionBinding.cpp --------------------------------*- C++ -*-===//
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

SBInstructionRef CreateSBInstruction() {
  return reinterpret_cast<SBInstructionRef>(new SBInstruction());
}

SBInstructionRef CloneSBInstruction(SBInstructionRef instance) {
  return reinterpret_cast<SBInstructionRef>(
      new SBInstruction(*reinterpret_cast<SBInstruction *>(instance)));
}

void DisposeSBInstruction(SBInstructionRef instance) {
  delete reinterpret_cast<SBInstruction *>(instance);
}

bool SBInstructionIsValid(SBInstructionRef instance) {
  SBInstruction *unwrapped = reinterpret_cast<SBInstruction *>(instance);
  return unwrapped->IsValid();
}

SBAddressRef SBInstructionGetAddress(SBInstructionRef instance) {
  SBInstruction *unwrapped = reinterpret_cast<SBInstruction *>(instance);
  return reinterpret_cast<SBAddressRef>(new SBAddress(unwrapped->GetAddress()));
}

const char *SBInstructionGetMnemonic(SBInstructionRef instance,
                                     SBTargetRef target) {
  SBInstruction *unwrapped = reinterpret_cast<SBInstruction *>(instance);
  return unwrapped->GetMnemonic(*reinterpret_cast<SBTarget *>(target));
}

const char *SBInstructionGetOperands(SBInstructionRef instance,
                                     SBTargetRef target) {
  SBInstruction *unwrapped = reinterpret_cast<SBInstruction *>(instance);
  return unwrapped->GetOperands(*reinterpret_cast<SBTarget *>(target));
}

const char *SBInstructionGetComment(SBInstructionRef instance,
                                    SBTargetRef target) {
  SBInstruction *unwrapped = reinterpret_cast<SBInstruction *>(instance);
  return unwrapped->GetComment(*reinterpret_cast<SBTarget *>(target));
}

lldb::InstructionControlFlowKind
SBInstructionGetControlFlowKind(SBInstructionRef instance, SBTargetRef target) {
  SBInstruction *unwrapped = reinterpret_cast<SBInstruction *>(instance);
  return unwrapped->GetControlFlowKind(*reinterpret_cast<SBTarget *>(target));
}

SBDataRef SBInstructionGetData(SBInstructionRef instance, SBTargetRef target) {
  SBInstruction *unwrapped = reinterpret_cast<SBInstruction *>(instance);
  return reinterpret_cast<SBDataRef>(
      new SBData(unwrapped->GetData(*reinterpret_cast<SBTarget *>(target))));
}

size_t SBInstructionGetByteSize(SBInstructionRef instance) {
  SBInstruction *unwrapped = reinterpret_cast<SBInstruction *>(instance);
  return unwrapped->GetByteSize();
}

bool SBInstructionDoesBranch(SBInstructionRef instance) {
  SBInstruction *unwrapped = reinterpret_cast<SBInstruction *>(instance);
  return unwrapped->DoesBranch();
}

bool SBInstructionHasDelaySlot(SBInstructionRef instance) {
  SBInstruction *unwrapped = reinterpret_cast<SBInstruction *>(instance);
  return unwrapped->HasDelaySlot();
}

void SBInstructionPrint(SBInstructionRef instance, SBFileRef out) {
  SBInstruction *unwrapped = reinterpret_cast<SBInstruction *>(instance);
  unwrapped->Print(*reinterpret_cast<SBFile *>(out));
}

bool SBInstructionGetDescription(SBInstructionRef instance,
                                 SBStreamRef description) {
  SBInstruction *unwrapped = reinterpret_cast<SBInstruction *>(instance);
  return unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description));
}

bool SBInstructionEmulateWithFrame(SBInstructionRef instance, SBFrameRef frame,
                                   uint32_t evaluate_options) {
  SBInstruction *unwrapped = reinterpret_cast<SBInstruction *>(instance);
  return unwrapped->EmulateWithFrame(*reinterpret_cast<SBFrame *>(frame),
                                     evaluate_options);
}

bool SBInstructionDumpEmulation(SBInstructionRef instance, const char *triple) {
  SBInstruction *unwrapped = reinterpret_cast<SBInstruction *>(instance);
  return unwrapped->DumpEmulation(triple);
}

bool SBInstructionTestEmulation(SBInstructionRef instance,
                                SBStreamRef output_stream,
                                const char *test_file) {
  SBInstruction *unwrapped = reinterpret_cast<SBInstruction *>(instance);
  return unwrapped->TestEmulation(*reinterpret_cast<SBStream *>(output_stream),
                                  test_file);
}

#ifdef __cplusplus
}
#endif
