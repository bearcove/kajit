//===-- SBTypeFormatBinding.cpp ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/LLDB.h"
#include "lldb/API/SBTypeFormat.h"
#include "lldb/Bindings/LLDBBinding.h"

using namespace lldb;

#ifdef __cplusplus
extern "C" {
#endif

SBTypeFormatRef CreateSBTypeFormat() {
  return reinterpret_cast<SBTypeFormatRef>(new SBTypeFormat());
}

SBTypeFormatRef CreateSBTypeFormat2(lldb::Format format, uint32_t options) {
  return reinterpret_cast<SBTypeFormatRef>(new SBTypeFormat(format, options));
}

SBTypeFormatRef CreateSBTypeFormat3(const char *type, uint32_t options) {
  return reinterpret_cast<SBTypeFormatRef>(new SBTypeFormat(type, options));
}

SBTypeFormatRef CloneSBTypeFormat(SBTypeFormatRef instance) {
  return reinterpret_cast<SBTypeFormatRef>(
      new SBTypeFormat(*reinterpret_cast<SBTypeFormat *>(instance)));
}

void DisposeSBTypeFormat(SBTypeFormatRef instance) {
  delete reinterpret_cast<SBTypeFormat *>(instance);
}

bool SBTypeFormatIsValid(SBTypeFormatRef instance) {
  SBTypeFormat *unwrapped = reinterpret_cast<SBTypeFormat *>(instance);
  return unwrapped->IsValid();
}

enum lldb::Format SBTypeFormatGetFormat(SBTypeFormatRef instance) {
  SBTypeFormat *unwrapped = reinterpret_cast<SBTypeFormat *>(instance);
  return unwrapped->GetFormat();
}

const char *SBTypeFormatGetTypeName(SBTypeFormatRef instance) {
  SBTypeFormat *unwrapped = reinterpret_cast<SBTypeFormat *>(instance);
  return unwrapped->GetTypeName();
}

uint32_t SBTypeFormatGetOptions(SBTypeFormatRef instance) {
  SBTypeFormat *unwrapped = reinterpret_cast<SBTypeFormat *>(instance);
  return unwrapped->GetOptions();
}

void SBTypeFormatSetFormat(SBTypeFormatRef instance, enum lldb::Format var0) {
  SBTypeFormat *unwrapped = reinterpret_cast<SBTypeFormat *>(instance);
  unwrapped->SetFormat(var0);
}

void SBTypeFormatSetTypeName(SBTypeFormatRef instance, const char *var0) {
  SBTypeFormat *unwrapped = reinterpret_cast<SBTypeFormat *>(instance);
  unwrapped->SetTypeName(var0);
}

void SBTypeFormatSetOptions(SBTypeFormatRef instance, uint32_t var0) {
  SBTypeFormat *unwrapped = reinterpret_cast<SBTypeFormat *>(instance);
  unwrapped->SetOptions(var0);
}

bool SBTypeFormatGetDescription(SBTypeFormatRef instance,
                                SBStreamRef description,
                                lldb::DescriptionLevel description_level) {
  SBTypeFormat *unwrapped = reinterpret_cast<SBTypeFormat *>(instance);
  return unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description),
                                   description_level);
}

bool SBTypeFormatIsEqualTo(SBTypeFormatRef instance, SBTypeFormatRef rhs) {
  SBTypeFormat *unwrapped = reinterpret_cast<SBTypeFormat *>(instance);
  return unwrapped->IsEqualTo(*reinterpret_cast<SBTypeFormat *>(rhs));
}

#ifdef __cplusplus
}
#endif
