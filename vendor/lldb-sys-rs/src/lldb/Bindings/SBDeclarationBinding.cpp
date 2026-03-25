//===-- SBDeclarationBinding.cpp --------------------------------*- C++ -*-===//
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

SBDeclarationRef CreateSBDeclaration() {
  return reinterpret_cast<SBDeclarationRef>(new SBDeclaration());
}

SBDeclarationRef CloneSBDeclaration(SBDeclarationRef instance) {
  return reinterpret_cast<SBDeclarationRef>(
      new SBDeclaration(*reinterpret_cast<SBDeclaration *>(instance)));
}

void DisposeSBDeclaration(SBDeclarationRef instance) {
  delete reinterpret_cast<SBDeclaration *>(instance);
}

bool SBDeclarationIsValid(SBDeclarationRef instance) {
  SBDeclaration *unwrapped = reinterpret_cast<SBDeclaration *>(instance);
  return unwrapped->IsValid();
}

SBFileSpecRef SBDeclarationGetFileSpec(SBDeclarationRef instance) {
  SBDeclaration *unwrapped = reinterpret_cast<SBDeclaration *>(instance);
  return reinterpret_cast<SBFileSpecRef>(
      new SBFileSpec(unwrapped->GetFileSpec()));
}

uint32_t SBDeclarationGetLine(SBDeclarationRef instance) {
  SBDeclaration *unwrapped = reinterpret_cast<SBDeclaration *>(instance);
  return unwrapped->GetLine();
}

uint32_t SBDeclarationGetColumn(SBDeclarationRef instance) {
  SBDeclaration *unwrapped = reinterpret_cast<SBDeclaration *>(instance);
  return unwrapped->GetColumn();
}

void SBDeclarationSetFileSpec(SBDeclarationRef instance,
                              SBFileSpecRef filespec) {
  SBDeclaration *unwrapped = reinterpret_cast<SBDeclaration *>(instance);
  unwrapped->SetFileSpec(*reinterpret_cast<SBFileSpec *>(filespec));
}

void SBDeclarationSetLine(SBDeclarationRef instance, uint32_t line) {
  SBDeclaration *unwrapped = reinterpret_cast<SBDeclaration *>(instance);
  unwrapped->SetLine(line);
}

void SBDeclarationSetColumn(SBDeclarationRef instance, uint32_t column) {
  SBDeclaration *unwrapped = reinterpret_cast<SBDeclaration *>(instance);
  unwrapped->SetColumn(column);
}

bool SBDeclarationGetDescription(SBDeclarationRef instance,
                                 SBStreamRef description) {
  SBDeclaration *unwrapped = reinterpret_cast<SBDeclaration *>(instance);
  return unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description));
}

#ifdef __cplusplus
}
#endif
