//===-- SBTypeSyntheticBinding.h --------------- ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBTypeSyntheticBinding_h_
#define LLDB_SBTypeSyntheticBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBTypeSyntheticRef CreateSBTypeSynthetic(void);

LLDB_API SBTypeSyntheticRef
SBTypeSyntheticCreateWithClassName(const char *data, uint32_t options);

LLDB_API SBTypeSyntheticRef
SBTypeSyntheticCreateWithScriptCode(const char *data, uint32_t options);

LLDB_API SBTypeSyntheticRef CloneSBTypeSynthetic(SBTypeSyntheticRef instance);

LLDB_API void DisposeSBTypeSynthetic(SBTypeSyntheticRef instance);

LLDB_API bool SBTypeSyntheticIsValid(SBTypeSyntheticRef instance);

LLDB_API bool SBTypeSyntheticIsClassCode(SBTypeSyntheticRef instance);

LLDB_API bool SBTypeSyntheticIsClassName(SBTypeSyntheticRef instance);

LLDB_API const char *SBTypeSyntheticGetData(SBTypeSyntheticRef instance);

LLDB_API void SBTypeSyntheticSetClassName(SBTypeSyntheticRef instance,
                                          const char *data);

LLDB_API void SBTypeSyntheticSetClassCode(SBTypeSyntheticRef instance,
                                          const char *data);

LLDB_API uint32_t SBTypeSyntheticGetOptions(SBTypeSyntheticRef instance);

LLDB_API void SBTypeSyntheticSetOptions(SBTypeSyntheticRef instance, uint32_t);

LLDB_API bool SBTypeSyntheticGetDescription(SBTypeSyntheticRef instance,
                                            SBStreamRef description,
                                            ENUM(DescriptionLevel)
                                                description_level);

LLDB_API bool SBTypeSyntheticIsEqualTo(SBTypeSyntheticRef instance,
                                       SBTypeSyntheticRef rhs);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBTypeSyntheticBinding_h_
