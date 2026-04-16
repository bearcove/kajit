//===-- SBTypeSyntheticBinding.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/LLDB.h"
#include "lldb/API/SBTypeSynthetic.h"
#include "lldb/Bindings/LLDBBinding.h"

using namespace lldb;

#ifdef __cplusplus
extern "C" {
#endif

SBTypeSyntheticRef CreateSBTypeSynthetic() {
  return reinterpret_cast<SBTypeSyntheticRef>(new SBTypeSynthetic());
}

SBTypeSyntheticRef SBTypeSyntheticCreateWithClassName(const char *data,
                                                      uint32_t options) {
  return reinterpret_cast<SBTypeSyntheticRef>(new SBTypeSynthetic(
      lldb::SBTypeSynthetic::CreateWithClassName(data, options)));
}

SBTypeSyntheticRef SBTypeSyntheticCreateWithScriptCode(const char *data,
                                                       uint32_t options) {
  return reinterpret_cast<SBTypeSyntheticRef>(new SBTypeSynthetic(
      lldb::SBTypeSynthetic::CreateWithScriptCode(data, options)));
}

SBTypeSyntheticRef CloneSBTypeSynthetic(SBTypeSyntheticRef instance) {
  return reinterpret_cast<SBTypeSyntheticRef>(
      new SBTypeSynthetic(*reinterpret_cast<SBTypeSynthetic *>(instance)));
}

void DisposeSBTypeSynthetic(SBTypeSyntheticRef instance) {
  delete reinterpret_cast<SBTypeSynthetic *>(instance);
}

bool SBTypeSyntheticIsValid(SBTypeSyntheticRef instance) {
  SBTypeSynthetic *unwrapped = reinterpret_cast<SBTypeSynthetic *>(instance);
  return unwrapped->IsValid();
}

bool SBTypeSyntheticIsClassCode(SBTypeSyntheticRef instance) {
  SBTypeSynthetic *unwrapped = reinterpret_cast<SBTypeSynthetic *>(instance);
  return unwrapped->IsClassCode();
}

bool SBTypeSyntheticIsClassName(SBTypeSyntheticRef instance) {
  SBTypeSynthetic *unwrapped = reinterpret_cast<SBTypeSynthetic *>(instance);
  return unwrapped->IsClassName();
}

const char *SBTypeSyntheticGetData(SBTypeSyntheticRef instance) {
  SBTypeSynthetic *unwrapped = reinterpret_cast<SBTypeSynthetic *>(instance);
  return unwrapped->GetData();
}

void SBTypeSyntheticSetClassName(SBTypeSyntheticRef instance,
                                 const char *data) {
  SBTypeSynthetic *unwrapped = reinterpret_cast<SBTypeSynthetic *>(instance);
  unwrapped->SetClassName(data);
}

void SBTypeSyntheticSetClassCode(SBTypeSyntheticRef instance,
                                 const char *data) {
  SBTypeSynthetic *unwrapped = reinterpret_cast<SBTypeSynthetic *>(instance);
  unwrapped->SetClassCode(data);
}

uint32_t SBTypeSyntheticGetOptions(SBTypeSyntheticRef instance) {
  SBTypeSynthetic *unwrapped = reinterpret_cast<SBTypeSynthetic *>(instance);
  return unwrapped->GetOptions();
}

void SBTypeSyntheticSetOptions(SBTypeSyntheticRef instance, uint32_t var0) {
  SBTypeSynthetic *unwrapped = reinterpret_cast<SBTypeSynthetic *>(instance);
  unwrapped->SetOptions(var0);
}

bool SBTypeSyntheticGetDescription(SBTypeSyntheticRef instance,
                                   SBStreamRef description,
                                   lldb::DescriptionLevel description_level) {
  SBTypeSynthetic *unwrapped = reinterpret_cast<SBTypeSynthetic *>(instance);
  return unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description),
                                   description_level);
}

bool SBTypeSyntheticIsEqualTo(SBTypeSyntheticRef instance,
                              SBTypeSyntheticRef rhs) {
  SBTypeSynthetic *unwrapped = reinterpret_cast<SBTypeSynthetic *>(instance);
  return unwrapped->IsEqualTo(*reinterpret_cast<SBTypeSynthetic *>(rhs));
}

#ifdef __cplusplus
}
#endif
