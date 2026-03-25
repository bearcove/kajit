//===-- SBStreamBinding.cpp -------------------------------------*- C++ -*-===//
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

SBStreamRef CreateSBStream() {
  return reinterpret_cast<SBStreamRef>(new SBStream());
}

void DisposeSBStream(SBStreamRef instance) {
  delete reinterpret_cast<SBStream *>(instance);
}

bool SBStreamIsValid(SBStreamRef instance) {
  SBStream *unwrapped = reinterpret_cast<SBStream *>(instance);
  return unwrapped->IsValid();
}

const char *SBStreamGetData(SBStreamRef instance) {
  SBStream *unwrapped = reinterpret_cast<SBStream *>(instance);
  return unwrapped->GetData();
}

size_t SBStreamGetSize(SBStreamRef instance) {
  SBStream *unwrapped = reinterpret_cast<SBStream *>(instance);
  return unwrapped->GetSize();
}

void SBStreamPrintf(SBStreamRef instance, const char *format, ...) {
  if (!format)
    return;

  SBStream *unwrapped = reinterpret_cast<SBStream *>(instance);
  va_list args;
  va_start(args, format);
  unwrapped->Printf(format, args);
  va_end(args);
}

void SBStreamPrint(SBStreamRef instance, const char *str) {
  SBStream *unwrapped = reinterpret_cast<SBStream *>(instance);
  unwrapped->Print(str);
}

void SBStreamRedirectToFile(SBStreamRef instance, const char *path,
                            bool append) {
  SBStream *unwrapped = reinterpret_cast<SBStream *>(instance);
  unwrapped->RedirectToFile(path, append);
}

void SBStreamRedirectToFile2(SBStreamRef instance, SBFileRef file) {
  SBStream *unwrapped = reinterpret_cast<SBStream *>(instance);
  unwrapped->RedirectToFile(*reinterpret_cast<SBFile *>(file));
}

void SBStreamRedirectToFileHandle(SBStreamRef instance, FILE *fh,
                                  bool transfer_fh_ownership) {
  SBStream *unwrapped = reinterpret_cast<SBStream *>(instance);
  unwrapped->RedirectToFileHandle(reinterpret_cast<FILE *>(fh),
                                  transfer_fh_ownership);
}

void SBStreamRedirectToFileDescriptor(SBStreamRef instance, int fd,
                                      bool transfer_fh_ownership) {
  SBStream *unwrapped = reinterpret_cast<SBStream *>(instance);
  unwrapped->RedirectToFileDescriptor(fd, transfer_fh_ownership);
}

void SBStreamClear(SBStreamRef instance) {
  SBStream *unwrapped = reinterpret_cast<SBStream *>(instance);
  unwrapped->Clear();
}

#ifdef __cplusplus
}
#endif
