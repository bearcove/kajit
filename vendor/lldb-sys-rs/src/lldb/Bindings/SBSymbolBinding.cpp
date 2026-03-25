//===-- SBSymbolBinding.cpp -------------------------------------*- C++ -*-===//
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

SBSymbolRef CreateSBSymbol() {
  return reinterpret_cast<SBSymbolRef>(new SBSymbol());
}

SBSymbolRef CloneSBSymbol(SBSymbolRef instance) {
  return reinterpret_cast<SBSymbolRef>(
      new SBSymbol(*reinterpret_cast<SBSymbol *>(instance)));
}

void DisposeSBSymbol(SBSymbolRef instance) {
  delete reinterpret_cast<SBSymbol *>(instance);
}

bool SBSymbolIsValid(SBSymbolRef instance) {
  SBSymbol *unwrapped = reinterpret_cast<SBSymbol *>(instance);
  return unwrapped->IsValid();
}

const char *SBSymbolGetName(SBSymbolRef instance) {
  SBSymbol *unwrapped = reinterpret_cast<SBSymbol *>(instance);
  return unwrapped->GetName();
}

const char *SBSymbolGetDisplayName(SBSymbolRef instance) {
  SBSymbol *unwrapped = reinterpret_cast<SBSymbol *>(instance);
  return unwrapped->GetDisplayName();
}

const char *SBSymbolGetMangledName(SBSymbolRef instance) {
  SBSymbol *unwrapped = reinterpret_cast<SBSymbol *>(instance);
  return unwrapped->GetMangledName();
}

SBInstructionListRef SBSymbolGetInstructions(SBSymbolRef instance,
                                             SBTargetRef target) {
  SBSymbol *unwrapped = reinterpret_cast<SBSymbol *>(instance);
  return reinterpret_cast<SBInstructionListRef>(new SBInstructionList(
      unwrapped->GetInstructions(*reinterpret_cast<SBTarget *>(target))));
}

SBInstructionListRef SBSymbolGetInstructions2(SBSymbolRef instance,
                                              SBTargetRef target,
                                              const char *flavor_string) {
  SBSymbol *unwrapped = reinterpret_cast<SBSymbol *>(instance);
  return reinterpret_cast<SBInstructionListRef>(
      new SBInstructionList(unwrapped->GetInstructions(
          *reinterpret_cast<SBTarget *>(target), flavor_string)));
}

SBAddressRef SBSymbolGetStartAddress(SBSymbolRef instance) {
  SBSymbol *unwrapped = reinterpret_cast<SBSymbol *>(instance);
  return reinterpret_cast<SBAddressRef>(
      new SBAddress(unwrapped->GetStartAddress()));
}

SBAddressRef SBSymbolGetEndAddress(SBSymbolRef instance) {
  SBSymbol *unwrapped = reinterpret_cast<SBSymbol *>(instance);
  return reinterpret_cast<SBAddressRef>(
      new SBAddress(unwrapped->GetEndAddress()));
}

uint32_t SBSymbolGetPrologueByteSize(SBSymbolRef instance) {
  SBSymbol *unwrapped = reinterpret_cast<SBSymbol *>(instance);
  return unwrapped->GetPrologueByteSize();
}

enum lldb::SymbolType SBSymbolGetType(SBSymbolRef instance) {
  SBSymbol *unwrapped = reinterpret_cast<SBSymbol *>(instance);
  return unwrapped->GetType();
}

bool SBSymbolGetDescription(SBSymbolRef instance, SBStreamRef description) {
  SBSymbol *unwrapped = reinterpret_cast<SBSymbol *>(instance);
  return unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description));
}

bool SBSymbolIsExternal(SBSymbolRef instance) {
  SBSymbol *unwrapped = reinterpret_cast<SBSymbol *>(instance);
  return unwrapped->IsExternal();
}

bool SBSymbolIsSynthetic(SBSymbolRef instance) {
  SBSymbol *unwrapped = reinterpret_cast<SBSymbol *>(instance);
  return unwrapped->IsSynthetic();
}

#ifdef __cplusplus
}
#endif
