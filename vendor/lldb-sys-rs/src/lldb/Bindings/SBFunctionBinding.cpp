//===-- SBFunctionBinding.cpp -----------------------------------*- C++ -*-===//
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

SBFunctionRef CreateSBFunction() {
  return reinterpret_cast<SBFunctionRef>(new SBFunction());
}

SBFunctionRef CloneSBFunction(SBFunctionRef instance) {
  return reinterpret_cast<SBFunctionRef>(
      new SBFunction(*reinterpret_cast<SBFunction *>(instance)));
}

void DisposeSBFunction(SBFunctionRef instance) {
  delete reinterpret_cast<SBFunction *>(instance);
}

bool SBFunctionIsValid(SBFunctionRef instance) {
  SBFunction *unwrapped = reinterpret_cast<SBFunction *>(instance);
  return unwrapped->IsValid();
}

const char *SBFunctionGetName(SBFunctionRef instance) {
  SBFunction *unwrapped = reinterpret_cast<SBFunction *>(instance);
  return unwrapped->GetName();
}

const char *SBFunctionGetDisplayName(SBFunctionRef instance) {
  SBFunction *unwrapped = reinterpret_cast<SBFunction *>(instance);
  return unwrapped->GetDisplayName();
}

const char *SBFunctionGetMangledName(SBFunctionRef instance) {
  SBFunction *unwrapped = reinterpret_cast<SBFunction *>(instance);
  return unwrapped->GetMangledName();
}

SBInstructionListRef SBFunctionGetInstructions(SBFunctionRef instance,
                                               SBTargetRef target) {
  SBFunction *unwrapped = reinterpret_cast<SBFunction *>(instance);
  return reinterpret_cast<SBInstructionListRef>(new SBInstructionList(
      unwrapped->GetInstructions(*reinterpret_cast<SBTarget *>(target))));
}

SBInstructionListRef SBFunctionGetInstructions2(SBFunctionRef instance,
                                                SBTargetRef target,
                                                const char *flavor) {
  SBFunction *unwrapped = reinterpret_cast<SBFunction *>(instance);
  return reinterpret_cast<SBInstructionListRef>(
      new SBInstructionList(unwrapped->GetInstructions(
          *reinterpret_cast<SBTarget *>(target), flavor)));
}

SBAddressRef SBFunctionGetStartAddress(SBFunctionRef instance) {
  SBFunction *unwrapped = reinterpret_cast<SBFunction *>(instance);
  return reinterpret_cast<SBAddressRef>(
      new SBAddress(unwrapped->GetStartAddress()));
}

SBAddressRef SBFunctionGetEndAddress(SBFunctionRef instance) {
  SBFunction *unwrapped = reinterpret_cast<SBFunction *>(instance);
  return reinterpret_cast<SBAddressRef>(
      new SBAddress(unwrapped->GetEndAddress()));
}

uint32_t SBFunctionGetPrologueByteSize(SBFunctionRef instance) {
  SBFunction *unwrapped = reinterpret_cast<SBFunction *>(instance);
  return unwrapped->GetPrologueByteSize();
}

SBTypeRef SBFunctionGetType(SBFunctionRef instance) {
  SBFunction *unwrapped = reinterpret_cast<SBFunction *>(instance);
  return reinterpret_cast<SBTypeRef>(new SBType(unwrapped->GetType()));
}

SBBlockRef SBFunctionGetBlock(SBFunctionRef instance) {
  SBFunction *unwrapped = reinterpret_cast<SBFunction *>(instance);
  return reinterpret_cast<SBBlockRef>(new SBBlock(unwrapped->GetBlock()));
}

lldb::LanguageType SBFunctionGetLanguage(SBFunctionRef instance) {
  SBFunction *unwrapped = reinterpret_cast<SBFunction *>(instance);
  return unwrapped->GetLanguage();
}

bool SBFunctionGetIsOptimized(SBFunctionRef instance) {
  SBFunction *unwrapped = reinterpret_cast<SBFunction *>(instance);
  return unwrapped->GetIsOptimized();
}

bool SBFunctionGetDescription(SBFunctionRef instance, SBStreamRef description) {
  SBFunction *unwrapped = reinterpret_cast<SBFunction *>(instance);
  return unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description));
}

#ifdef __cplusplus
}
#endif
