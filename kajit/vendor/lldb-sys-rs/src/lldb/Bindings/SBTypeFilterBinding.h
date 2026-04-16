//===-- SBTypeFilterBinding.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBTypeFilterBinding_h_
#define LLDB_SBTypeFilterBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBTypeFilterRef CreateSBTypeFilter(void);

LLDB_API SBTypeFilterRef CreateSBTypeFilter2(uint32_t options);

LLDB_API SBTypeFilterRef CloneSBTypeFilter(SBTypeFilterRef instance);

LLDB_API void DisposeSBTypeFilter(SBTypeFilterRef instance);

LLDB_API bool SBTypeFilterIsValid(SBTypeFilterRef instance);

LLDB_API uint32_t
SBTypeFilterGetNumberOfExpressionPaths(SBTypeFilterRef instance);

LLDB_API const char *
SBTypeFilterGetExpressionPathAtIndex(SBTypeFilterRef instance, uint32_t i);

LLDB_API bool SBTypeFilterReplaceExpressionPathAtIndex(SBTypeFilterRef instance,
                                                       uint32_t i,
                                                       const char *item);

LLDB_API void SBTypeFilterAppendExpressionPath(SBTypeFilterRef instance,
                                               const char *item);

LLDB_API void SBTypeFilterClear(SBTypeFilterRef instance);

LLDB_API uint32_t SBTypeFilterGetOptions(SBTypeFilterRef instance);

LLDB_API void SBTypeFilterSetOptions(SBTypeFilterRef instance, uint32_t);

LLDB_API bool SBTypeFilterGetDescription(SBTypeFilterRef instance,
                                         SBStreamRef description,
                                         ENUM(DescriptionLevel)
                                             description_level);

LLDB_API bool SBTypeFilterIsEqualTo(SBTypeFilterRef instance,
                                    SBTypeFilterRef rhs);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBTypeFilterBinding_h_
