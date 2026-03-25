//===-- SBCommandReturnObjectBinding.cpp ------------------------*- C++ -*-===//
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

SBCommandReturnObjectRef CreateSBCommandReturnObject() {
  return reinterpret_cast<SBCommandReturnObjectRef>(
      new SBCommandReturnObject());
}

SBCommandReturnObjectRef
CloneSBCommandReturnObject(SBCommandReturnObjectRef instance) {
  return reinterpret_cast<SBCommandReturnObjectRef>(new SBCommandReturnObject(
      *reinterpret_cast<SBCommandReturnObject *>(instance)));
}

void DisposeSBCommandReturnObject(SBCommandReturnObjectRef instance) {
  delete reinterpret_cast<SBCommandReturnObject *>(instance);
}

bool SBCommandReturnObjectIsValid(SBCommandReturnObjectRef instance) {
  SBCommandReturnObject *unwrapped =
      reinterpret_cast<SBCommandReturnObject *>(instance);
  return unwrapped->IsValid();
}

const char *SBCommandReturnObjectGetOutput(SBCommandReturnObjectRef instance) {
  SBCommandReturnObject *unwrapped =
      reinterpret_cast<SBCommandReturnObject *>(instance);
  return unwrapped->GetOutput();
}

const char *SBCommandReturnObjectGetError(SBCommandReturnObjectRef instance) {
  SBCommandReturnObject *unwrapped =
      reinterpret_cast<SBCommandReturnObject *>(instance);
  return unwrapped->GetError();
}

size_t SBCommandReturnObjectPutOutput(SBCommandReturnObjectRef instance,
                                      SBFileRef file) {
  SBCommandReturnObject *unwrapped =
      reinterpret_cast<SBCommandReturnObject *>(instance);
  return unwrapped->PutOutput(*reinterpret_cast<SBFile *>(file));
}

size_t SBCommandReturnObjectGetOutputSize(SBCommandReturnObjectRef instance) {
  SBCommandReturnObject *unwrapped =
      reinterpret_cast<SBCommandReturnObject *>(instance);
  return unwrapped->GetOutputSize();
}

size_t SBCommandReturnObjectGetErrorSize(SBCommandReturnObjectRef instance) {
  SBCommandReturnObject *unwrapped =
      reinterpret_cast<SBCommandReturnObject *>(instance);
  return unwrapped->GetErrorSize();
}

size_t SBCommandReturnObjectPutError(SBCommandReturnObjectRef instance,
                                     SBFileRef file) {
  SBCommandReturnObject *unwrapped =
      reinterpret_cast<SBCommandReturnObject *>(instance);
  return unwrapped->PutError(*reinterpret_cast<SBFile *>(file));
}

void SBCommandReturnObjectClear(SBCommandReturnObjectRef instance) {
  SBCommandReturnObject *unwrapped =
      reinterpret_cast<SBCommandReturnObject *>(instance);
  unwrapped->Clear();
}

enum lldb::ReturnStatus
SBCommandReturnObjectGetStatus(SBCommandReturnObjectRef instance) {
  SBCommandReturnObject *unwrapped =
      reinterpret_cast<SBCommandReturnObject *>(instance);
  return unwrapped->GetStatus();
}

void SBCommandReturnObjectSetStatus(SBCommandReturnObjectRef instance,
                                    lldb::ReturnStatus status) {
  SBCommandReturnObject *unwrapped =
      reinterpret_cast<SBCommandReturnObject *>(instance);
  unwrapped->SetStatus(status);
}

bool SBCommandReturnObjectSucceeded(SBCommandReturnObjectRef instance) {
  SBCommandReturnObject *unwrapped =
      reinterpret_cast<SBCommandReturnObject *>(instance);
  return unwrapped->Succeeded();
}

bool SBCommandReturnObjectHasResult(SBCommandReturnObjectRef instance) {
  SBCommandReturnObject *unwrapped =
      reinterpret_cast<SBCommandReturnObject *>(instance);
  return unwrapped->HasResult();
}

void SBCommandReturnObjectAppendMessage(SBCommandReturnObjectRef instance,
                                        const char *message) {
  SBCommandReturnObject *unwrapped =
      reinterpret_cast<SBCommandReturnObject *>(instance);
  unwrapped->AppendMessage(message);
}

void SBCommandReturnObjectAppendWarning(SBCommandReturnObjectRef instance,
                                        const char *message) {
  SBCommandReturnObject *unwrapped =
      reinterpret_cast<SBCommandReturnObject *>(instance);
  unwrapped->AppendWarning(message);
}

bool SBCommandReturnObjectGetDescription(SBCommandReturnObjectRef instance,
                                         SBStreamRef description) {
  SBCommandReturnObject *unwrapped =
      reinterpret_cast<SBCommandReturnObject *>(instance);
  return unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description));
}

void SBCommandReturnObjectSetImmediateOutputFile(
    SBCommandReturnObjectRef instance, SBFileRef file) {
  SBCommandReturnObject *unwrapped =
      reinterpret_cast<SBCommandReturnObject *>(instance);
  unwrapped->SetImmediateOutputFile(*reinterpret_cast<SBFile *>(file));
}

void SBCommandReturnObjectSetImmediateErrorFile(
    SBCommandReturnObjectRef instance, SBFileRef file) {
  SBCommandReturnObject *unwrapped =
      reinterpret_cast<SBCommandReturnObject *>(instance);
  unwrapped->SetImmediateErrorFile(*reinterpret_cast<SBFile *>(file));
}

void SBCommandReturnObjectPutCString(SBCommandReturnObjectRef instance,
                                     const char *string, int len) {
  SBCommandReturnObject *unwrapped =
      reinterpret_cast<SBCommandReturnObject *>(instance);
  unwrapped->PutCString(string, len);
}

size_t SBCommandReturnObjectPrintf(SBCommandReturnObjectRef instance,
                                   const char *format, ...) {
  if (!format)
    return 0;

  size_t retVal;
  SBCommandReturnObject *unwrapped =
      reinterpret_cast<SBCommandReturnObject *>(instance);

  va_list args;
  va_start(args, format);
  retVal = unwrapped->Printf(format, args);
  va_end(args);

  return retVal;
}

const char *SBCommandReturnObjectGetOutput2(SBCommandReturnObjectRef instance,
                                            bool only_if_no_immediate) {
  SBCommandReturnObject *unwrapped =
      reinterpret_cast<SBCommandReturnObject *>(instance);
  return unwrapped->GetOutput(only_if_no_immediate);
}

const char *SBCommandReturnObjectGetError2(SBCommandReturnObjectRef instance,
                                           bool only_if_no_immediate) {
  SBCommandReturnObject *unwrapped =
      reinterpret_cast<SBCommandReturnObject *>(instance);
  return unwrapped->GetError(only_if_no_immediate);
}

void SBCommandReturnObjectSetError(SBCommandReturnObjectRef instance,
                                   SBErrorRef error,
                                   const char *fallback_error_cstr) {
  SBCommandReturnObject *unwrapped =
      reinterpret_cast<SBCommandReturnObject *>(instance);
  unwrapped->SetError(*reinterpret_cast<SBError *>(error), fallback_error_cstr);
}

void SBCommandReturnObjectSetError2(SBCommandReturnObjectRef instance,
                                    const char *error_cstr) {
  SBCommandReturnObject *unwrapped =
      reinterpret_cast<SBCommandReturnObject *>(instance);
  unwrapped->SetError(error_cstr);
}

#ifdef __cplusplus
}
#endif
