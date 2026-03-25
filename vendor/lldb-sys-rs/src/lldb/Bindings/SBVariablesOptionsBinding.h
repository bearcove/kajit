//===-- SBVariablesOptionsBinding.h -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBVariablesOptionsBinding_h_
#define LLDB_SBVariablesOptionsBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBVariablesOptionsRef CreateSBVariablesOptions(void);

LLDB_API SBVariablesOptionsRef
CreateSBVariablesOptions2(SBVariablesOptionsRef options);

LLDB_API SBVariablesOptionsRef
CloneSBVariablesOptions(SBVariablesOptionsRef instance);

LLDB_API void DisposeSBVariablesOptions(SBVariablesOptionsRef instance);

LLDB_API bool SBVariablesOptionsIsValid(SBVariablesOptionsRef instance);

LLDB_API bool
SBVariablesOptionsGetIncludeArguments(SBVariablesOptionsRef instance);

LLDB_API void
SBVariablesOptionsSetIncludeArguments(SBVariablesOptionsRef instance, bool);

LLDB_API bool
SBVariablesOptionsGetIncludeRecognizedArguments(SBVariablesOptionsRef instance,
                                                SBTargetRef);

LLDB_API void
SBVariablesOptionsSetIncludeRecognizedArguments(SBVariablesOptionsRef instance,
                                                bool);

LLDB_API bool
SBVariablesOptionsGetIncludeLocals(SBVariablesOptionsRef instance);

LLDB_API void SBVariablesOptionsSetIncludeLocals(SBVariablesOptionsRef instance,
                                                 bool);

LLDB_API bool
SBVariablesOptionsGetIncludeStatics(SBVariablesOptionsRef instance);

LLDB_API void
SBVariablesOptionsSetIncludeStatics(SBVariablesOptionsRef instance, bool);

LLDB_API bool SBVariablesOptionsGetInScopeOnly(SBVariablesOptionsRef instance);

LLDB_API void SBVariablesOptionsSetInScopeOnly(SBVariablesOptionsRef instance,
                                               bool);

LLDB_API bool SBVariablesOptionsGetIncludeRuntimeSupportValues(
    SBVariablesOptionsRef instance);

LLDB_API void
SBVariablesOptionsSetIncludeRuntimeSupportValues(SBVariablesOptionsRef instance,
                                                 bool);

LLDB_API ENUM(DynamicValueType)
    SBVariablesOptionsGetUseDynamic(SBVariablesOptionsRef instance);

LLDB_API void SBVariablesOptionsSetUseDynamic(SBVariablesOptionsRef instance,
                                              ENUM(DynamicValueType));

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBVariablesOptionsBinding_h_
