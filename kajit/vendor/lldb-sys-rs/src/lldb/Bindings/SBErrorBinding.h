//===-- SBErrorBinding.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBErrorBinding_h_
#define LLDB_SBErrorBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBErrorRef CreateSBError(void);

LLDB_API SBErrorRef CloneSBError(SBErrorRef instance);

LLDB_API void DisposeSBError(SBErrorRef instance);

LLDB_API const char *SBErrorGetCString(SBErrorRef instance);

LLDB_API void SBErrorClear(SBErrorRef instance);

LLDB_API bool SBErrorFail(SBErrorRef instance);

LLDB_API bool SBErrorSuccess(SBErrorRef instance);

LLDB_API uint32_t SBErrorGetError(SBErrorRef instance);

LLDB_API ENUM(ErrorType) SBErrorGetType(SBErrorRef instance);

LLDB_API void SBErrorSetError(SBErrorRef instance, uint32_t err,
                              ENUM(ErrorType) type);

LLDB_API void SBErrorSetErrorToErrno(SBErrorRef instance);

LLDB_API void SBErrorSetErrorToGenericError(SBErrorRef instance);

LLDB_API void SBErrorSetErrorString(SBErrorRef instance, const char *err_str);

LLDB_API int SBErrorSetErrorStringWithFormat(SBErrorRef instance,
                                             const char *format, ...);

LLDB_API bool SBErrorIsValid(SBErrorRef instance);

LLDB_API bool SBErrorGetDescription(SBErrorRef instance,
                                    SBStreamRef description);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBErrorBinding_h_
