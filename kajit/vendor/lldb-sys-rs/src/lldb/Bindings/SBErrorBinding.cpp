//===-- SBErrorBinding.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/LLDB.h"
#include "lldb/Bindings/LLDBBinding.h"
#include <stdarg.h>

using namespace lldb;

#ifdef __cplusplus
extern "C" {
#endif

SBErrorRef CreateSBError() {
  return reinterpret_cast<SBErrorRef>(new SBError());
}

SBErrorRef CloneSBError(SBErrorRef instance) {
  return reinterpret_cast<SBErrorRef>(
      new SBError(*reinterpret_cast<SBError *>(instance)));
}

void DisposeSBError(SBErrorRef instance) {
  delete reinterpret_cast<SBError *>(instance);
}

const char *SBErrorGetCString(SBErrorRef instance) {
  SBError *unwrapped = reinterpret_cast<SBError *>(instance);
  return unwrapped->GetCString();
}

void SBErrorClear(SBErrorRef instance) {
  SBError *unwrapped = reinterpret_cast<SBError *>(instance);
  unwrapped->Clear();
}

bool SBErrorFail(SBErrorRef instance) {
  SBError *unwrapped = reinterpret_cast<SBError *>(instance);
  return unwrapped->Fail();
}

bool SBErrorSuccess(SBErrorRef instance) {
  SBError *unwrapped = reinterpret_cast<SBError *>(instance);
  return unwrapped->Success();
}

uint32_t SBErrorGetError(SBErrorRef instance) {
  SBError *unwrapped = reinterpret_cast<SBError *>(instance);
  return unwrapped->GetError();
}

enum lldb::ErrorType SBErrorGetType(SBErrorRef instance) {
  SBError *unwrapped = reinterpret_cast<SBError *>(instance);
  return unwrapped->GetType();
}

void SBErrorSetError(SBErrorRef instance, uint32_t err, lldb::ErrorType type) {
  SBError *unwrapped = reinterpret_cast<SBError *>(instance);
  unwrapped->SetError(err, type);
}

void SBErrorSetErrorToErrno(SBErrorRef instance) {
  SBError *unwrapped = reinterpret_cast<SBError *>(instance);
  unwrapped->SetErrorToErrno();
}

void SBErrorSetErrorToGenericError(SBErrorRef instance) {
  SBError *unwrapped = reinterpret_cast<SBError *>(instance);
  unwrapped->SetErrorToGenericError();
}

void SBErrorSetErrorString(SBErrorRef instance, const char *err_str) {
  SBError *unwrapped = reinterpret_cast<SBError *>(instance);
  unwrapped->SetErrorString(err_str);
}

int SBErrorSetErrorStringWithFormat(SBErrorRef instance, const char *format,
                                    ...) {
  if (!format)
    return 0;

  int retVal;
  SBError *unwrapped = reinterpret_cast<SBError *>(instance);

  va_list args;
  va_start(args, format);
  retVal = unwrapped->SetErrorStringWithFormat(format, args);
  va_end(args);
  return retVal;
}

bool SBErrorIsValid(SBErrorRef instance) {
  SBError *unwrapped = reinterpret_cast<SBError *>(instance);
  return unwrapped->IsValid();
}

bool SBErrorGetDescription(SBErrorRef instance, SBStreamRef description) {
  SBError *unwrapped = reinterpret_cast<SBError *>(instance);
  return unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description));
}

#ifdef __cplusplus
}
#endif
