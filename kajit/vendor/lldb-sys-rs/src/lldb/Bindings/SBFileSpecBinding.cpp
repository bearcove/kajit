//===-- SBFileSpecBinding.cpp -----------------------------------*- C++ -*-===//
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

SBFileSpecRef CreateSBFileSpec() {
  return reinterpret_cast<SBFileSpecRef>(new SBFileSpec());
}

SBFileSpecRef CreateSBFileSpec2(const char *path) {
  return reinterpret_cast<SBFileSpecRef>(new SBFileSpec(path));
}

SBFileSpecRef CreateSBFileSpec3(const char *path, bool resolve) {
  return reinterpret_cast<SBFileSpecRef>(new SBFileSpec(path, resolve));
}

SBFileSpecRef CloneSBFileSpec(SBFileSpecRef instance) {
  return reinterpret_cast<SBFileSpecRef>(
      new SBFileSpec(*reinterpret_cast<SBFileSpec *>(instance)));
}

void DisposeSBFileSpec(SBFileSpecRef instance) {
  delete reinterpret_cast<SBFileSpec *>(instance);
}

bool SBFileSpecIsValid(SBFileSpecRef instance) {
  SBFileSpec *unwrapped = reinterpret_cast<SBFileSpec *>(instance);
  return unwrapped->IsValid();
}

bool SBFileSpecExists(SBFileSpecRef instance) {
  SBFileSpec *unwrapped = reinterpret_cast<SBFileSpec *>(instance);
  return unwrapped->Exists();
}

bool SBFileSpecResolveExecutableLocation(SBFileSpecRef instance) {
  SBFileSpec *unwrapped = reinterpret_cast<SBFileSpec *>(instance);
  return unwrapped->ResolveExecutableLocation();
}

const char *SBFileSpecGetFilename(SBFileSpecRef instance) {
  SBFileSpec *unwrapped = reinterpret_cast<SBFileSpec *>(instance);
  return unwrapped->GetFilename();
}

const char *SBFileSpecGetDirectory(SBFileSpecRef instance) {
  SBFileSpec *unwrapped = reinterpret_cast<SBFileSpec *>(instance);
  return unwrapped->GetDirectory();
}

void SBFileSpecSetFilename(SBFileSpecRef instance, const char *filename) {
  SBFileSpec *unwrapped = reinterpret_cast<SBFileSpec *>(instance);
  unwrapped->SetFilename(filename);
}

void SBFileSpecSetDirectory(SBFileSpecRef instance, const char *directory) {
  SBFileSpec *unwrapped = reinterpret_cast<SBFileSpec *>(instance);
  unwrapped->SetDirectory(directory);
}

uint32_t SBFileSpecGetPath(SBFileSpecRef instance, char *dst_path,
                           size_t dst_len) {
  SBFileSpec *unwrapped = reinterpret_cast<SBFileSpec *>(instance);
  return unwrapped->GetPath(dst_path, dst_len);
}

int SBFileSpecResolvePath(const char *src_path, char *dst_path,
                          size_t dst_len) {
  return lldb::SBFileSpec::ResolvePath(src_path, dst_path, dst_len);
}

bool SBFileSpecGetDescription(SBFileSpecRef instance, SBStreamRef description) {
  SBFileSpec *unwrapped = reinterpret_cast<SBFileSpec *>(instance);
  return unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description));
}

void SBFileSpecAppendPathComponent(SBFileSpecRef instance,
                                   const char *file_or_directory) {
  SBFileSpec *unwrapped = reinterpret_cast<SBFileSpec *>(instance);
  unwrapped->AppendPathComponent(file_or_directory);
}

#ifdef __cplusplus
}
#endif
