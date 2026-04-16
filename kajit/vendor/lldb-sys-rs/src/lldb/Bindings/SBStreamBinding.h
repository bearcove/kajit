//===-- SBStreamBinding.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBStreamBinding_h_
#define LLDB_SBStreamBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBStreamRef CreateSBStream(void);

LLDB_API void DisposeSBStream(SBStreamRef instance);

LLDB_API bool SBStreamIsValid(SBStreamRef instance);

LLDB_API const char *SBStreamGetData(SBStreamRef instance);

LLDB_API size_t SBStreamGetSize(SBStreamRef instance);

LLDB_API void SBStreamPrintf(SBStreamRef instance, const char *format, ...);

LLDB_API void SBStreamPrint(SBStreamRef instance, const char *str);

LLDB_API void SBStreamRedirectToFile(SBStreamRef instance, const char *path,
                                     bool append);

LLDB_API void SBStreamRedirectToFile2(SBStreamRef instance, SBFileRef file);

LLDB_API void SBStreamRedirectToFileHandle(SBStreamRef instance, FILE *fh,
                                           bool transfer_fh_ownership);

LLDB_API void SBStreamRedirectToFileDescriptor(SBStreamRef instance, int fd,
                                               bool transfer_fh_ownership);

LLDB_API void SBStreamClear(SBStreamRef instance);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBStreamBinding_h_
