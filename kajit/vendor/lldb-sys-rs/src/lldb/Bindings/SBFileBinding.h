//===-- SBFileBinding.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBFileBinding_h_
#define LLDB_SBFileBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBFileRef CreateSBFile(void);

LLDB_API SBFileRef CreateSBFile2(FILE *file, bool transfer_ownership);

LLDB_API SBFileRef CreateSBFile3(int fd, const char *mode,
                                 bool transfer_ownership);

LLDB_API SBFileRef CloneSBFile(SBFileRef instance);

LLDB_API void DisposeSBFile(SBFileRef instance);

LLDB_API bool SBFileIsValid(SBFileRef instance);

LLDB_API SBErrorRef SBFileRead(SBFileRef instance, uint8_t *buf,
                               size_t num_bytes, size_t *bytes_read);

LLDB_API SBErrorRef SBFileWrite(SBFileRef instance, const uint8_t *buf,
                                size_t num_bytes, size_t *bytes_written);

LLDB_API SBErrorRef SBFileFlush(SBFileRef instance);

LLDB_API SBErrorRef SBFileClose(SBFileRef instance);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBFileBinding_h_
