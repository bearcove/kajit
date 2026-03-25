//===-- SBCompileUnitBinding.cpp --------------------------------*- C++ -*-===//
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

SBCompileUnitRef CreateSBCompileUnit() {
  return reinterpret_cast<SBCompileUnitRef>(new SBCompileUnit());
}

SBCompileUnitRef CloneSBCompileUnit(SBCompileUnitRef instance) {
  return reinterpret_cast<SBCompileUnitRef>(
      new SBCompileUnit(*reinterpret_cast<SBCompileUnit *>(instance)));
}

void DisposeSBCompileUnit(SBCompileUnitRef instance) {
  delete reinterpret_cast<SBCompileUnit *>(instance);
}

bool SBCompileUnitIsValid(SBCompileUnitRef instance) {
  SBCompileUnit *unwrapped = reinterpret_cast<SBCompileUnit *>(instance);
  return unwrapped->IsValid();
}

SBFileSpecRef SBCompileUnitGetFileSpec(SBCompileUnitRef instance) {
  SBCompileUnit *unwrapped = reinterpret_cast<SBCompileUnit *>(instance);
  return reinterpret_cast<SBFileSpecRef>(
      new SBFileSpec(unwrapped->GetFileSpec()));
}

uint32_t SBCompileUnitGetNumLineEntries(SBCompileUnitRef instance) {
  SBCompileUnit *unwrapped = reinterpret_cast<SBCompileUnit *>(instance);
  return unwrapped->GetNumLineEntries();
}

SBLineEntryRef SBCompileUnitGetLineEntryAtIndex(SBCompileUnitRef instance,
                                                uint32_t idx) {
  SBCompileUnit *unwrapped = reinterpret_cast<SBCompileUnit *>(instance);
  return reinterpret_cast<SBLineEntryRef>(
      new SBLineEntry(unwrapped->GetLineEntryAtIndex(idx)));
}

uint32_t SBCompileUnitFindLineEntryIndex(SBCompileUnitRef instance,
                                         uint32_t start_idx, uint32_t line,
                                         SBFileSpecRef inline_file_spec) {
  SBCompileUnit *unwrapped = reinterpret_cast<SBCompileUnit *>(instance);
  return unwrapped->FindLineEntryIndex(
      start_idx, line, reinterpret_cast<SBFileSpec *>(inline_file_spec));
}

uint32_t SBCompileUnitFindLineEntryIndex2(SBCompileUnitRef instance,
                                          uint32_t start_idx, uint32_t line,
                                          SBFileSpecRef inline_file_spec,
                                          bool exact) {
  SBCompileUnit *unwrapped = reinterpret_cast<SBCompileUnit *>(instance);
  return unwrapped->FindLineEntryIndex(
      start_idx, line, reinterpret_cast<SBFileSpec *>(inline_file_spec), exact);
}

SBFileSpecRef SBCompileUnitGetSupportFileAtIndex(SBCompileUnitRef instance,
                                                 uint32_t idx) {
  SBCompileUnit *unwrapped = reinterpret_cast<SBCompileUnit *>(instance);
  return reinterpret_cast<SBFileSpecRef>(
      new SBFileSpec(unwrapped->GetSupportFileAtIndex(idx)));
}

uint32_t SBCompileUnitGetNumSupportFiles(SBCompileUnitRef instance) {
  SBCompileUnit *unwrapped = reinterpret_cast<SBCompileUnit *>(instance);
  return unwrapped->GetNumSupportFiles();
}

uint32_t SBCompileUnitFindSupportFileIndex(SBCompileUnitRef instance,
                                           uint32_t start_idx,
                                           SBFileSpecRef sb_file, bool full) {
  SBCompileUnit *unwrapped = reinterpret_cast<SBCompileUnit *>(instance);
  return unwrapped->FindSupportFileIndex(
      start_idx, *reinterpret_cast<SBFileSpec *>(sb_file), full);
}

SBTypeListRef SBCompileUnitGetTypes(SBCompileUnitRef instance,
                                    uint32_t type_mask) {
  SBCompileUnit *unwrapped = reinterpret_cast<SBCompileUnit *>(instance);
  return reinterpret_cast<SBTypeListRef>(
      new SBTypeList(unwrapped->GetTypes(type_mask)));
}

enum lldb::LanguageType SBCompileUnitGetLanguage(SBCompileUnitRef instance) {
  SBCompileUnit *unwrapped = reinterpret_cast<SBCompileUnit *>(instance);
  return unwrapped->GetLanguage();
}

bool SBCompileUnitGetDescription(SBCompileUnitRef instance,
                                 SBStreamRef description) {
  SBCompileUnit *unwrapped = reinterpret_cast<SBCompileUnit *>(instance);
  return unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description));
}

#ifdef __cplusplus
}
#endif
