//===-- SBStructuredDataBinding.h -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBStructuredDataBinding_h_
#define LLDB_SBStructuredDataBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBStructuredDataRef CreateSBStructuredData(void);

LLDB_API SBStructuredDataRef
CloneSBStructuredData(SBStructuredDataRef instance);

LLDB_API void DisposeSBStructuredData(SBStructuredDataRef instance);

LLDB_API bool SBStructuredDataIsValid(SBStructuredDataRef instance);

LLDB_API void SBStructuredDataClear(SBStructuredDataRef instance);

LLDB_API SBErrorRef SBStructuredDataSetFromJSON(SBStructuredDataRef instance,
                                                SBStreamRef stream);

LLDB_API SBErrorRef SBStructuredDataSetFromJSON2(SBStructuredDataRef instance,
                                                 const char *json);

LLDB_API SBErrorRef SBStructuredDataGetAsJSON(SBStructuredDataRef instance,
                                              SBStreamRef stream);

LLDB_API SBErrorRef SBStructuredDataGetDescription(SBStructuredDataRef instance,
                                                   SBStreamRef stream);

LLDB_API StructuredDataType SBStructureDataGetType(SBStructureDataRef instance);

LLDB_API size_t SBStructuredDataGetSize(SBStructuredDataRef instance);

LLDB_API bool SBStructuredDataGetKeys(SBStructuredDataRef instance,
                                      SBStringListRef keys);

LLDB_API SBStructuredDataRef
SBStructuredDataGetValueForKey(SBStructureDataRef instance, const char *key);

LLDB_API SBStructuredDataRef
SBStructuredDataGetItemAtIndex(SBStructuredDataRef instance, size_t idx);

LLDB_API uint64_t SBStructuredDataGetIntegerValue(SBStructuredDataRef instance,
                                                  uint64_t fail_value);

LLDB_API double SBStructuredDataGetFloatValue(SBStructuredDataRef instance,
                                              double fail_value);

LLDB_API bool SBStructuredDataGetBooleanValue(SBStructuredDataRef instance,
                                              bool fail_value);

LLDB_API size_t SBStructuredDataGetStringValue(SBStructuredData instance,
                                               char *dst, size_t dst_len);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBStructuredDataBinding_h_
