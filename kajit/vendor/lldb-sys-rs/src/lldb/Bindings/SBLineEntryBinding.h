//===-- SBLineEntryBinding.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBLineEntryBinding_h_
#define LLDB_SBLineEntryBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBLineEntryRef CreateSBLineEntry(void);

LLDB_API SBLineEntryRef CloneSBLineEntry(SBLineEntryRef instance);

LLDB_API void DisposeSBLineEntry(SBLineEntryRef instance);

LLDB_API SBAddressRef SBLineEntryGetStartAddress(SBLineEntryRef instance);

LLDB_API SBAddressRef SBLineEntryGetEndAddress(SBLineEntryRef instance);

LLDB_API bool SBLineEntryIsValid(SBLineEntryRef instance);

LLDB_API SBFileSpecRef SBLineEntryGetFileSpec(SBLineEntryRef instance);

LLDB_API uint32_t SBLineEntryGetLine(SBLineEntryRef instance);

LLDB_API uint32_t SBLineEntryGetColumn(SBLineEntryRef instance);

LLDB_API void SBLineEntrySetFileSpec(SBLineEntryRef instance,
                                     SBFileSpecRef filespec);

LLDB_API void SBLineEntrySetLine(SBLineEntryRef instance, uint32_t line);

LLDB_API void SBLineEntrySetColumn(SBLineEntryRef instance, uint32_t column);

LLDB_API bool SBLineEntryGetDescription(SBLineEntryRef instance,
                                        SBStreamRef description);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBLineEntryBinding_h_
