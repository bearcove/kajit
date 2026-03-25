//===-- SBFileBinding.cpp -----------------------------------*- C++ -*-===//
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

SBFileRef CreateSBFile() { return reinterpret_cast<SBFileRef>(new SBFile()); }

SBFileRef CreateSBFile2(FILE *file, bool transfer_ownership) {
  return reinterpret_cast<SBFileRef>(new SBFile(file, transfer_ownership));
}

SBFileRef CreateSBFile3(int fd, const char *mode, bool transfer_ownership) {
  return reinterpret_cast<SBFileRef>(new SBFile(fd, mode, transfer_ownership));
}

SBFileRef CloneSBFile(SBFileRef instance) {
  return reinterpret_cast<SBFileRef>(
      new SBFile(*reinterpret_cast<SBFile *>(instance)));
}

void DisposeSBFile(SBFileRef instance) {
  delete reinterpret_cast<SBFile *>(instance);
}

bool SBFileIsValid(SBFileRef instance) {
  SBFile *unwrapped = reinterpret_cast<SBFile *>(instance);
  return unwrapped->IsValid();
}

SBErrorRef SBFileRead(SBFileRef instance, uint8_t *buf, size_t num_bytes,
                      size_t *bytes_read) {
  SBFile *unwrapped = reinterpret_cast<SBFile *>(instance);
  return reinterpret_cast<SBErrorRef>(
      new SBError(unwrapped->Read(buf, num_bytes, bytes_read)));
}

SBErrorRef SBFileWrite(SBFileRef instance, const uint8_t *buf, size_t num_bytes,
                       size_t *bytes_written) {
  SBFile *unwrapped = reinterpret_cast<SBFile *>(instance);
  return reinterpret_cast<SBErrorRef>(
      new SBError(unwrapped->Write(buf, num_bytes, bytes_written)));
}

SBErrorRef SBFileFlush(SBFileRef instance) {
  SBFile *unwrapped = reinterpret_cast<SBFile *>(instance);
  return reinterpret_cast<SBErrorRef>(new SBError(unwrapped->Flush()));
}

SBErrorRef SBFileClose(SBFileRef instance) {
  SBFile *unwrapped = reinterpret_cast<SBFile *>(instance);
  return reinterpret_cast<SBErrorRef>(new SBError(unwrapped->Close()));
}

#ifdef __cplusplus
}
#endif
