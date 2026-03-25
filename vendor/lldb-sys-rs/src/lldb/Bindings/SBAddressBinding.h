//===-- SBAddressBinding.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBAddressBinding_h_
#define LLDB_SBAddressBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBAddressRef CreateSBAddress(void);

LLDB_API SBAddressRef CreateSBAddress2(SBSectionRef section,
                                       lldb_addr_t offset);

LLDB_API SBAddressRef CreateSBAddress3(lldb_addr_t load_addr,
                                       SBTargetRef target);

LLDB_API SBAddressRef CloneSBAddress(SBAddressRef instance);

LLDB_API void DisposeSBAddress(SBAddressRef instance);

LLDB_API bool SBAddressIsValid(SBAddressRef instance);

LLDB_API void SBAddressClear(SBAddressRef instance);

LLDB_API lldb_addr_t SBAddressGetFileAddress(SBAddressRef instance);

LLDB_API lldb_addr_t SBAddressGetLoadAddress(SBAddressRef instance,
                                             SBTargetRef target);

LLDB_API void SBAddressSetAddress(SBAddressRef instance, SBSectionRef section,
                                  lldb_addr_t offset);

LLDB_API void SBAddressSetLoadAddress(SBAddressRef instance,
                                      lldb_addr_t load_addr,
                                      SBTargetRef target);

LLDB_API bool SBAddressOffsetAddress(SBAddressRef instance, lldb_addr_t offset);

LLDB_API bool SBAddressGetDescription(SBAddressRef instance,
                                      SBStreamRef description);

LLDB_API SBSymbolContextRef SBAddressGetSymbolContext(SBAddressRef instance,
                                                      uint32_t resolve_scope);

LLDB_API SBSectionRef SBAddressGetSection(SBAddressRef instance);

LLDB_API lldb_addr_t SBAddressGetOffset(SBAddressRef instance);

LLDB_API SBModuleRef SBAddressGetModule(SBAddressRef instance);

LLDB_API SBCompileUnitRef SBAddressGetCompileUnit(SBAddressRef instance);

LLDB_API SBFunctionRef SBAddressGetFunction(SBAddressRef instance);

LLDB_API SBBlockRef SBAddressGetBlock(SBAddressRef instance);

LLDB_API SBSymbolRef SBAddressGetSymbol(SBAddressRef instance);

LLDB_API SBLineEntryRef SBAddressGetLineEntry(SBAddressRef instance);

LLDB_API bool SBAddressIsEqual(SBAddressRef instance, SBAddressRef other);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBAddressBinding_h_
