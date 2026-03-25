//===-- SBSymbolContextBinding.cpp ------------------------------*- C++ -*-===//
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

SBSymbolContextRef CreateSBSymbolContext() {
  return reinterpret_cast<SBSymbolContextRef>(new SBSymbolContext());
}

SBSymbolContextRef CloneSBSymbolContext(SBSymbolContextRef instance) {
  return reinterpret_cast<SBSymbolContextRef>(
      new SBSymbolContext(*reinterpret_cast<SBSymbolContext *>(instance)));
}

void DisposeSBSymbolContext(SBSymbolContextRef instance) {
  delete reinterpret_cast<SBSymbolContext *>(instance);
}

bool SBSymbolContextIsValid(SBSymbolContextRef instance) {
  SBSymbolContext *unwrapped = reinterpret_cast<SBSymbolContext *>(instance);
  return unwrapped->IsValid();
}

SBModuleRef SBSymbolContextGetModule(SBSymbolContextRef instance) {
  SBSymbolContext *unwrapped = reinterpret_cast<SBSymbolContext *>(instance);
  return reinterpret_cast<SBModuleRef>(new SBModule(unwrapped->GetModule()));
}

SBCompileUnitRef SBSymbolContextGetCompileUnit(SBSymbolContextRef instance) {
  SBSymbolContext *unwrapped = reinterpret_cast<SBSymbolContext *>(instance);
  return reinterpret_cast<SBCompileUnitRef>(
      new SBCompileUnit(unwrapped->GetCompileUnit()));
}

SBFunctionRef SBSymbolContextGetFunction(SBSymbolContextRef instance) {
  SBSymbolContext *unwrapped = reinterpret_cast<SBSymbolContext *>(instance);
  return reinterpret_cast<SBFunctionRef>(
      new SBFunction(unwrapped->GetFunction()));
}

SBBlockRef SBSymbolContextGetBlock(SBSymbolContextRef instance) {
  SBSymbolContext *unwrapped = reinterpret_cast<SBSymbolContext *>(instance);
  return reinterpret_cast<SBBlockRef>(new SBBlock(unwrapped->GetBlock()));
}

SBLineEntryRef SBSymbolContextGetLineEntry(SBSymbolContextRef instance) {
  SBSymbolContext *unwrapped = reinterpret_cast<SBSymbolContext *>(instance);
  return reinterpret_cast<SBLineEntryRef>(
      new SBLineEntry(unwrapped->GetLineEntry()));
}

SBSymbolRef SBSymbolContextGetSymbol(SBSymbolContextRef instance) {
  SBSymbolContext *unwrapped = reinterpret_cast<SBSymbolContext *>(instance);
  return reinterpret_cast<SBSymbolRef>(new SBSymbol(unwrapped->GetSymbol()));
}

void SBSymbolContextSetModule(SBSymbolContextRef instance, SBModuleRef module) {
  SBSymbolContext *unwrapped = reinterpret_cast<SBSymbolContext *>(instance);
  unwrapped->SetModule(*reinterpret_cast<SBModule *>(module));
}

void SBSymbolContextSetCompileUnit(SBSymbolContextRef instance,
                                   SBCompileUnitRef compile_unit) {
  SBSymbolContext *unwrapped = reinterpret_cast<SBSymbolContext *>(instance);
  unwrapped->SetCompileUnit(*reinterpret_cast<SBCompileUnit *>(compile_unit));
}

void SBSymbolContextSetFunction(SBSymbolContextRef instance,
                                SBFunctionRef function) {
  SBSymbolContext *unwrapped = reinterpret_cast<SBSymbolContext *>(instance);
  unwrapped->SetFunction(*reinterpret_cast<SBFunction *>(function));
}

void SBSymbolContextSetBlock(SBSymbolContextRef instance, SBBlockRef block) {
  SBSymbolContext *unwrapped = reinterpret_cast<SBSymbolContext *>(instance);
  unwrapped->SetBlock(*reinterpret_cast<SBBlock *>(block));
}

void SBSymbolContextSetLineEntry(SBSymbolContextRef instance,
                                 SBLineEntryRef line_entry) {
  SBSymbolContext *unwrapped = reinterpret_cast<SBSymbolContext *>(instance);
  unwrapped->SetLineEntry(*reinterpret_cast<SBLineEntry *>(line_entry));
}

void SBSymbolContextSetSymbol(SBSymbolContextRef instance, SBSymbolRef symbol) {
  SBSymbolContext *unwrapped = reinterpret_cast<SBSymbolContext *>(instance);
  unwrapped->SetSymbol(*reinterpret_cast<SBSymbol *>(symbol));
}

SBSymbolContextRef
SBSymbolContextGetParentOfInlinedScope(SBSymbolContextRef instance,
                                       SBAddressRef curr_frame_pc,
                                       SBAddressRef parent_frame_addr) {
  SBSymbolContext *unwrapped = reinterpret_cast<SBSymbolContext *>(instance);
  return reinterpret_cast<SBSymbolContextRef>(
      new SBSymbolContext(unwrapped->GetParentOfInlinedScope(
          *reinterpret_cast<SBAddress *>(curr_frame_pc),
          *reinterpret_cast<SBAddress *>(parent_frame_addr))));
}

bool SBSymbolContextGetDescription(SBSymbolContextRef instance,
                                   SBStreamRef description) {
  SBSymbolContext *unwrapped = reinterpret_cast<SBSymbolContext *>(instance);
  return unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description));
}

#ifdef __cplusplus
}
#endif
