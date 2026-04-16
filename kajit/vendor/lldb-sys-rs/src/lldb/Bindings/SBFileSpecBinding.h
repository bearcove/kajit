//===-- SBFileSpecBinding.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBFileSpecBinding_h_
#define LLDB_SBFileSpecBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBFileSpecRef CreateSBFileSpec(void);

LLDB_API SBFileSpecRef CreateSBFileSpec2(const char *path);

LLDB_API SBFileSpecRef CreateSBFileSpec3(const char *path, bool resolve);

LLDB_API SBFileSpecRef CloneSBFileSpec(SBFileSpecRef instance);

LLDB_API void DisposeSBFileSpec(SBFileSpecRef instance);

LLDB_API bool SBFileSpecIsValid(SBFileSpecRef instance);

LLDB_API bool SBFileSpecExists(SBFileSpecRef instance);

LLDB_API bool SBFileSpecResolveExecutableLocation(SBFileSpecRef instance);

LLDB_API const char *SBFileSpecGetFilename(SBFileSpecRef instance);

LLDB_API const char *SBFileSpecGetDirectory(SBFileSpecRef instance);

LLDB_API void SBFileSpecSetFilename(SBFileSpecRef instance,
                                    const char *filename);

LLDB_API void SBFileSpecSetDirectory(SBFileSpecRef instance,
                                     const char *directory);

LLDB_API uint32_t SBFileSpecGetPath(SBFileSpecRef instance, char *dst_path,
                                    size_t dst_len);

LLDB_API int SBFileSpecResolvePath(const char *src_path, char *dst_path,
                                   size_t dst_len);

LLDB_API bool SBFileSpecGetDescription(SBFileSpecRef instance,
                                       SBStreamRef description);

LLDB_API void SBFileSpecAppendPathComponent(SBFileSpecRef instance,
                                            const char *file_or_directory);
#ifdef __cplusplus
}
#endif

#endif // LLDB_SBFileSpecBinding_h_
