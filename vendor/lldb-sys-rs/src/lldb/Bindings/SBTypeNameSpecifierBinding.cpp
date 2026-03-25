//===-- SBTypeNameSpecifierBinding.cpp --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/LLDB.h"
#include "lldb/API/SBTypeNameSpecifier.h"
#include "lldb/Bindings/LLDBBinding.h"

using namespace lldb;

#ifdef __cplusplus
extern "C" {
#endif

SBTypeNameSpecifierRef CreateSBTypeNameSpecifier() {
  return reinterpret_cast<SBTypeNameSpecifierRef>(new SBTypeNameSpecifier());
}

SBTypeNameSpecifierRef CreateSBTypeNameSpecifier2(const char *name,
                                                  bool is_regex) {
  return reinterpret_cast<SBTypeNameSpecifierRef>(
      new SBTypeNameSpecifier(name, is_regex));
}

SBTypeNameSpecifierRef CreateSBTypeNameSpecifier3(SBTypeRef type) {
  return reinterpret_cast<SBTypeNameSpecifierRef>(
      new SBTypeNameSpecifier(*reinterpret_cast<SBType *>(type)));
}

SBTypeNameSpecifierRef
CloneSBTypeNameSpecifier(SBTypeNameSpecifierRef instance) {
  return reinterpret_cast<SBTypeNameSpecifierRef>(new SBTypeNameSpecifier(
      *reinterpret_cast<SBTypeNameSpecifier *>(instance)));
}

void DisposeSBTypeNameSpecifier(SBTypeNameSpecifierRef instance) {
  delete reinterpret_cast<SBTypeNameSpecifier *>(instance);
}

bool SBTypeNameSpecifierIsValid(SBTypeNameSpecifierRef instance) {
  SBTypeNameSpecifier *unwrapped =
      reinterpret_cast<SBTypeNameSpecifier *>(instance);
  return unwrapped->IsValid();
}

const char *SBTypeNameSpecifierGetName(SBTypeNameSpecifierRef instance) {
  SBTypeNameSpecifier *unwrapped =
      reinterpret_cast<SBTypeNameSpecifier *>(instance);
  return unwrapped->GetName();
}

SBTypeRef SBTypeNameSpecifierGetType(SBTypeNameSpecifierRef instance) {
  SBTypeNameSpecifier *unwrapped =
      reinterpret_cast<SBTypeNameSpecifier *>(instance);
  return reinterpret_cast<SBTypeRef>(new SBType(unwrapped->GetType()));
}

bool SBTypeNameSpecifierIsRegex(SBTypeNameSpecifierRef instance) {
  SBTypeNameSpecifier *unwrapped =
      reinterpret_cast<SBTypeNameSpecifier *>(instance);
  return unwrapped->IsRegex();
}

bool SBTypeNameSpecifierGetDescription(
    SBTypeNameSpecifierRef instance, SBStreamRef description,
    lldb::DescriptionLevel description_level) {
  SBTypeNameSpecifier *unwrapped =
      reinterpret_cast<SBTypeNameSpecifier *>(instance);
  return unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description),
                                   description_level);
}

bool SBTypeNameSpecifierIsEqualTo(SBTypeNameSpecifierRef instance,
                                  SBTypeNameSpecifierRef rhs) {
  SBTypeNameSpecifier *unwrapped =
      reinterpret_cast<SBTypeNameSpecifier *>(instance);
  return unwrapped->IsEqualTo(*reinterpret_cast<SBTypeNameSpecifier *>(rhs));
}

#ifdef __cplusplus
}
#endif
