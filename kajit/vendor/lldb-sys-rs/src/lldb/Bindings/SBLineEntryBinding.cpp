//===-- SBLineEntryBinding.cpp ----------------------------------*- C++ -*-===//
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

SBLineEntryRef CreateSBLineEntry() {
  return reinterpret_cast<SBLineEntryRef>(new SBLineEntry());
}

SBLineEntryRef CloneSBLineEntry(SBLineEntryRef instance) {
  return reinterpret_cast<SBLineEntryRef>(
      new SBLineEntry(*reinterpret_cast<SBLineEntry *>(instance)));
}

void DisposeSBLineEntry(SBLineEntryRef instance) {
  delete reinterpret_cast<SBLineEntry *>(instance);
}

SBAddressRef SBLineEntryGetStartAddress(SBLineEntryRef instance) {
  SBLineEntry *unwrapped = reinterpret_cast<SBLineEntry *>(instance);
  return reinterpret_cast<SBAddressRef>(
      new SBAddress(unwrapped->GetStartAddress()));
}

SBAddressRef SBLineEntryGetEndAddress(SBLineEntryRef instance) {
  SBLineEntry *unwrapped = reinterpret_cast<SBLineEntry *>(instance);
  return reinterpret_cast<SBAddressRef>(
      new SBAddress(unwrapped->GetEndAddress()));
}

bool SBLineEntryIsValid(SBLineEntryRef instance) {
  SBLineEntry *unwrapped = reinterpret_cast<SBLineEntry *>(instance);
  return unwrapped->IsValid();
}

SBFileSpecRef SBLineEntryGetFileSpec(SBLineEntryRef instance) {
  SBLineEntry *unwrapped = reinterpret_cast<SBLineEntry *>(instance);
  return reinterpret_cast<SBFileSpecRef>(
      new SBFileSpec(unwrapped->GetFileSpec()));
}

uint32_t SBLineEntryGetLine(SBLineEntryRef instance) {
  SBLineEntry *unwrapped = reinterpret_cast<SBLineEntry *>(instance);
  return unwrapped->GetLine();
}

uint32_t SBLineEntryGetColumn(SBLineEntryRef instance) {
  SBLineEntry *unwrapped = reinterpret_cast<SBLineEntry *>(instance);
  return unwrapped->GetColumn();
}

void SBLineEntrySetFileSpec(SBLineEntryRef instance, SBFileSpecRef filespec) {
  SBLineEntry *unwrapped = reinterpret_cast<SBLineEntry *>(instance);
  unwrapped->SetFileSpec(*reinterpret_cast<SBFileSpec *>(filespec));
}

void SBLineEntrySetLine(SBLineEntryRef instance, uint32_t line) {
  SBLineEntry *unwrapped = reinterpret_cast<SBLineEntry *>(instance);
  unwrapped->SetLine(line);
}

void SBLineEntrySetColumn(SBLineEntryRef instance, uint32_t column) {
  SBLineEntry *unwrapped = reinterpret_cast<SBLineEntry *>(instance);
  unwrapped->SetColumn(column);
}

bool SBLineEntryGetDescription(SBLineEntryRef instance,
                               SBStreamRef description) {
  SBLineEntry *unwrapped = reinterpret_cast<SBLineEntry *>(instance);
  return unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description));
}

#ifdef __cplusplus
}
#endif
