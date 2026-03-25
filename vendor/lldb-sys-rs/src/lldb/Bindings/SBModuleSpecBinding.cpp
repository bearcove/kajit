//===-- SBModuleSpecBinding.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/LLDB.h"
#include "lldb/API/SBModuleSpec.h"
#include "lldb/Bindings/LLDBBinding.h"

using namespace lldb;

#ifdef __cplusplus
extern "C" {
#endif

SBModuleSpecRef CreateSBModuleSpec() {
  return reinterpret_cast<SBModuleSpecRef>(new SBModuleSpec());
}

SBModuleSpecRef CloneSBModuleSpec(SBModuleSpecRef instance) {
  return reinterpret_cast<SBModuleSpecRef>(
      new SBModuleSpec(*reinterpret_cast<SBModuleSpec *>(instance)));
}

void DisposeSBModuleSpec(SBModuleSpecRef instance) {
  delete reinterpret_cast<SBModuleSpec *>(instance);
}

bool SBModuleSpecIsValid(SBModuleSpecRef instance) {
  SBModuleSpec *unwrapped = reinterpret_cast<SBModuleSpec *>(instance);
  return unwrapped->IsValid();
}

void SBModuleSpecClear(SBModuleSpecRef instance) {
  SBModuleSpec *unwrapped = reinterpret_cast<SBModuleSpec *>(instance);
  unwrapped->Clear();
}

SBFileSpecRef SBModuleSpecGetFileSpec(SBModuleSpecRef instance) {
  SBModuleSpec *unwrapped = reinterpret_cast<SBModuleSpec *>(instance);
  return reinterpret_cast<SBFileSpecRef>(
      new SBFileSpec(unwrapped->GetFileSpec()));
}

void SBModuleSpecSetFileSpec(SBModuleSpecRef instance, SBFileSpecRef fspec) {
  SBModuleSpec *unwrapped = reinterpret_cast<SBModuleSpec *>(instance);
  unwrapped->SetFileSpec(*reinterpret_cast<SBFileSpec *>(fspec));
}

SBFileSpecRef SBModuleSpecGetPlatformFileSpec(SBModuleSpecRef instance) {
  SBModuleSpec *unwrapped = reinterpret_cast<SBModuleSpec *>(instance);
  return reinterpret_cast<SBFileSpecRef>(
      new SBFileSpec(unwrapped->GetPlatformFileSpec()));
}

void SBModuleSpecSetPlatformFileSpec(SBModuleSpecRef instance,
                                     SBFileSpecRef fspec) {
  SBModuleSpec *unwrapped = reinterpret_cast<SBModuleSpec *>(instance);
  unwrapped->SetPlatformFileSpec(*reinterpret_cast<SBFileSpec *>(fspec));
}

SBFileSpecRef SBModuleSpecGetSymbolFileSpec(SBModuleSpecRef instance) {
  SBModuleSpec *unwrapped = reinterpret_cast<SBModuleSpec *>(instance);
  return reinterpret_cast<SBFileSpecRef>(
      new SBFileSpec(unwrapped->GetSymbolFileSpec()));
}

void SBModuleSpecSetSymbolFileSpec(SBModuleSpecRef instance,
                                   SBFileSpecRef fspec) {
  SBModuleSpec *unwrapped = reinterpret_cast<SBModuleSpec *>(instance);
  unwrapped->SetSymbolFileSpec(*reinterpret_cast<SBFileSpec *>(fspec));
}

const char *SBModuleSpecGetObjectName(SBModuleSpecRef instance) {
  SBModuleSpec *unwrapped = reinterpret_cast<SBModuleSpec *>(instance);
  return unwrapped->GetObjectName();
}

void SBModuleSpecSetObjectName(SBModuleSpecRef instance, const char *name) {
  SBModuleSpec *unwrapped = reinterpret_cast<SBModuleSpec *>(instance);
  unwrapped->SetObjectName(name);
}

const char *SBModuleSpecGetTriple(SBModuleSpecRef instance) {
  SBModuleSpec *unwrapped = reinterpret_cast<SBModuleSpec *>(instance);
  return unwrapped->GetTriple();
}

void SBModuleSpecSetTriple(SBModuleSpecRef instance, const char *triple) {
  SBModuleSpec *unwrapped = reinterpret_cast<SBModuleSpec *>(instance);
  unwrapped->SetTriple(triple);
}

const uint8_t *SBModuleSpecGetUUIDBytes(SBModuleSpecRef instance) {
  SBModuleSpec *unwrapped = reinterpret_cast<SBModuleSpec *>(instance);
  return unwrapped->GetUUIDBytes();
}

size_t SBModuleSpecGetUUIDLength(SBModuleSpecRef instance) {
  SBModuleSpec *unwrapped = reinterpret_cast<SBModuleSpec *>(instance);
  return unwrapped->GetUUIDLength();
}

bool SBModuleSpecSetUUIDBytes(SBModuleSpecRef instance, const uint8_t *uuid,
                              size_t uuid_len) {
  SBModuleSpec *unwrapped = reinterpret_cast<SBModuleSpec *>(instance);
  return unwrapped->SetUUIDBytes(uuid, uuid_len);
}

bool SBModuleSpecGetDescription(SBModuleSpecRef instance,
                                SBStreamRef description) {
  SBModuleSpec *unwrapped = reinterpret_cast<SBModuleSpec *>(instance);
  return unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description));
}

SBModuleSpecListRef CreateSBModuleSpecList() {
  return reinterpret_cast<SBModuleSpecListRef>(new SBModuleSpecList());
}

SBModuleSpecListRef CloneSBModuleSpecList(SBModuleSpecListRef instance) {
  return reinterpret_cast<SBModuleSpecListRef>(
      new SBModuleSpecList(*reinterpret_cast<SBModuleSpecList *>(instance)));
}

void DisposeSBModuleSpecList(SBModuleSpecListRef instance) {
  delete reinterpret_cast<SBModuleSpecList *>(instance);
}

SBModuleSpecListRef SBModuleSpecListGetModuleSpecifications(const char *path) {
  return reinterpret_cast<SBModuleSpecListRef>(new SBModuleSpecList(
      lldb::SBModuleSpecList::GetModuleSpecifications(path)));
}

void SBModuleSpecListAppend(SBModuleSpecListRef instance,
                            SBModuleSpecRef spec) {
  SBModuleSpecList *unwrapped = reinterpret_cast<SBModuleSpecList *>(instance);
  unwrapped->Append(*reinterpret_cast<SBModuleSpec *>(spec));
}

void SBModuleSpecListAppendList(SBModuleSpecListRef instance,
                                SBModuleSpecListRef spec_list) {
  SBModuleSpecList *unwrapped = reinterpret_cast<SBModuleSpecList *>(instance);
  unwrapped->Append(*reinterpret_cast<SBModuleSpecList *>(spec_list));
}

SBModuleSpecRef
SBModuleSpecListFindFirstMatchingSpec(SBModuleSpecListRef instance,
                                      SBModuleSpecRef match_spec) {
  SBModuleSpecList *unwrapped = reinterpret_cast<SBModuleSpecList *>(instance);
  return reinterpret_cast<SBModuleSpecRef>(
      new SBModuleSpec(unwrapped->FindFirstMatchingSpec(
          *reinterpret_cast<SBModuleSpec *>(match_spec))));
}

SBModuleSpecListRef
SBModuleSpecListFindMatchingSpecs(SBModuleSpecListRef instance,
                                  SBModuleSpecRef match_spec) {
  SBModuleSpecList *unwrapped = reinterpret_cast<SBModuleSpecList *>(instance);
  return reinterpret_cast<SBModuleSpecListRef>(
      new SBModuleSpecList(unwrapped->FindMatchingSpecs(
          *reinterpret_cast<SBModuleSpec *>(match_spec))));
}

size_t SBModuleSpecListGetSize(SBModuleSpecListRef instance) {
  SBModuleSpecList *unwrapped = reinterpret_cast<SBModuleSpecList *>(instance);
  return unwrapped->GetSize();
}

SBModuleSpecRef SBModuleSpecListGetSpecAtIndex(SBModuleSpecListRef instance,
                                               size_t i) {
  SBModuleSpecList *unwrapped = reinterpret_cast<SBModuleSpecList *>(instance);
  return reinterpret_cast<SBModuleSpecRef>(
      new SBModuleSpec(unwrapped->GetSpecAtIndex(i)));
}

bool SBModuleSpecListGetDescription(SBModuleSpecListRef instance,
                                    SBStreamRef description) {
  SBModuleSpecList *unwrapped = reinterpret_cast<SBModuleSpecList *>(instance);
  return unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description));
}

#ifdef __cplusplus
}
#endif
