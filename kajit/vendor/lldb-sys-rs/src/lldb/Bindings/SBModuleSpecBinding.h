//===-- SBModuleSpecBinding.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBModuleSpecBinding_h_
#define LLDB_SBModuleSpecBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBModuleSpecRef CreateSBModuleSpec(void);

LLDB_API SBModuleSpecRef CloneSBModuleSpec(SBModuleSpecRef instance);

LLDB_API void DisposeSBModuleSpec(SBModuleSpecRef instance);

LLDB_API bool SBModuleSpecIsValid(SBModuleSpecRef instance);

LLDB_API void SBModuleSpecClear(SBModuleSpecRef instance);

LLDB_API SBFileSpecRef SBModuleSpecGetFileSpec(SBModuleSpecRef instance);

LLDB_API void SBModuleSpecSetFileSpec(SBModuleSpecRef instance,
                                      SBFileSpecRef fspec);

LLDB_API SBFileSpecRef
SBModuleSpecGetPlatformFileSpec(SBModuleSpecRef instance);

LLDB_API void SBModuleSpecSetPlatformFileSpec(SBModuleSpecRef instance,
                                              SBFileSpecRef fspec);

LLDB_API SBFileSpecRef SBModuleSpecGetSymbolFileSpec(SBModuleSpecRef instance);

LLDB_API void SBModuleSpecSetSymbolFileSpec(SBModuleSpecRef instance,
                                            SBFileSpecRef fspec);

LLDB_API const char *SBModuleSpecGetObjectName(SBModuleSpecRef instance);

LLDB_API void SBModuleSpecSetObjectName(SBModuleSpecRef instance,
                                        const char *name);

LLDB_API const char *SBModuleSpecGetTriple(SBModuleSpecRef instance);

LLDB_API void SBModuleSpecSetTriple(SBModuleSpecRef instance,
                                    const char *triple);

LLDB_API const uint8_t *SBModuleSpecGetUUIDBytes(SBModuleSpecRef instance);

LLDB_API size_t SBModuleSpecGetUUIDLength(SBModuleSpecRef instance);

LLDB_API bool SBModuleSpecSetUUIDBytes(SBModuleSpecRef instance,
                                       const uint8_t *uuid, size_t uuid_len);

LLDB_API bool SBModuleSpecGetDescription(SBModuleSpecRef instance,
                                         SBStreamRef description);

LLDB_API SBModuleSpecListRef CreateSBModuleSpecList(void);

LLDB_API SBModuleSpecListRef
CloneSBModuleSpecList(SBModuleSpecListRef instance);

LLDB_API void DisposeSBModuleSpecList(SBModuleSpecListRef instance);

LLDB_API SBModuleSpecListRef
SBModuleSpecListGetModuleSpecifications(const char *path);

LLDB_API void SBModuleSpecListAppend(SBModuleSpecListRef instance,
                                     SBModuleSpecRef spec);

LLDB_API void SBModuleSpecListAppendList(SBModuleSpecListRef instance,
                                         SBModuleSpecListRef spec_list);

LLDB_API SBModuleSpecRef SBModuleSpecListFindFirstMatchingSpec(
    SBModuleSpecListRef instance, SBModuleSpecRef match_spec);

LLDB_API SBModuleSpecListRef SBModuleSpecListFindMatchingSpecs(
    SBModuleSpecListRef instance, SBModuleSpecRef match_spec);

LLDB_API size_t SBModuleSpecListGetSize(SBModuleSpecListRef instance);

LLDB_API SBModuleSpecRef
SBModuleSpecListGetSpecAtIndex(SBModuleSpecListRef instance, size_t i);

LLDB_API bool SBModuleSpecListGetDescription(SBModuleSpecListRef instance,
                                             SBStreamRef description);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBModuleSpecBinding_h_
