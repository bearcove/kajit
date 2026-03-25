//===-- SBCommandReturnObjectBinding.h --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBCommandReturnObjectBinding_h_
#define LLDB_SBCommandReturnObjectBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBCommandReturnObjectRef CreateSBCommandReturnObject(void);

LLDB_API SBCommandReturnObjectRef
CloneSBCommandReturnObject(SBCommandReturnObjectRef instance);

LLDB_API void DisposeSBCommandReturnObject(SBCommandReturnObjectRef instance);

LLDB_API bool SBCommandReturnObjectIsValid(SBCommandReturnObjectRef instance);

LLDB_API const char *
SBCommandReturnObjectGetOutput(SBCommandReturnObjectRef instance);

LLDB_API const char *
SBCommandReturnObjectGetError(SBCommandReturnObjectRef instance);

LLDB_API size_t SBCommandReturnObjectPutOutput(
    SBCommandReturnObjectRef instance, SBFileRef file);

LLDB_API size_t
SBCommandReturnObjectGetOutputSize(SBCommandReturnObjectRef instance);

LLDB_API size_t
SBCommandReturnObjectGetErrorSize(SBCommandReturnObjectRef instance);

LLDB_API size_t SBCommandReturnObjectPutError(SBCommandReturnObjectRef instance,
                                              SBFileRef file);

LLDB_API void SBCommandReturnObjectClear(SBCommandReturnObjectRef instance);

LLDB_API ENUM(ReturnStatus)
    SBCommandReturnObjectGetStatus(SBCommandReturnObjectRef instance);

LLDB_API void SBCommandReturnObjectSetStatus(SBCommandReturnObjectRef instance,
                                             ENUM(ReturnStatus) status);

LLDB_API bool SBCommandReturnObjectSucceeded(SBCommandReturnObjectRef instance);

LLDB_API bool SBCommandReturnObjectHasResult(SBCommandReturnObjectRef instance);

LLDB_API void
SBCommandReturnObjectAppendMessage(SBCommandReturnObjectRef instance,
                                   const char *message);

LLDB_API void
SBCommandReturnObjectAppendWarning(SBCommandReturnObjectRef instance,
                                   const char *message);

LLDB_API bool
SBCommandReturnObjectGetDescription(SBCommandReturnObjectRef instance,
                                    SBStreamRef description);

LLDB_API void
SBCommandReturnObjectSetImmediateOutputFile(SBCommandReturnObjectRef instance,
                                            SBFileRef file);

LLDB_API void
SBCommandReturnObjectSetImmediateErrorFile(SBCommandReturnObjectRef instance,
                                           SBFileRef file);

LLDB_API void SBCommandReturnObjectPutCString(SBCommandReturnObjectRef instance,
                                              const char *string, int len);

LLDB_API size_t SBCommandReturnObjectPrintf(SBCommandReturnObjectRef instance,
                                            const char *format, ...);

LLDB_API const char *
SBCommandReturnObjectGetOutput2(SBCommandReturnObjectRef instance,
                                bool only_if_no_immediate);

LLDB_API const char *
SBCommandReturnObjectGetError2(SBCommandReturnObjectRef instance,
                               bool only_if_no_immediate);

LLDB_API void SBCommandReturnObjectSetError(SBCommandReturnObjectRef instance,
                                            SBErrorRef error,
                                            const char *fallback_error_cstr);

LLDB_API void SBCommandReturnObjectSetError2(SBCommandReturnObjectRef instance,
                                             const char *error_cstr);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBCommandReturnObjectBinding_h_
