//===-- SBModuleBinding.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBModuleBinding_h_
#define LLDB_SBModuleBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBModuleRef CreateSBModule(void);

LLDB_API SBModuleRef CreateSBModule2(SBModuleSpecRef module_spec);

LLDB_API SBModuleRef CreateSBModule3(SBProcessRef process,
                                     lldb_addr_t header_addr);

LLDB_API SBModuleRef CloneSBModule(SBModuleRef instance);

LLDB_API void DisposeSBModule(SBModuleRef instance);

LLDB_API bool SBModuleIsValid(SBModuleRef instance);

LLDB_API void SBModuleClear(SBModuleRef instance);

LLDB_API bool SBModuleIsFileBacked(SBModuleRef instance);

LLDB_API SBFileSpecRef SBModuleGetFileSpec(SBModuleRef instance);

LLDB_API SBFileSpecRef SBModuleGetPlatformFileSpec(SBModuleRef instance);

LLDB_API bool SBModuleSetPlatformFileSpec(SBModuleRef instance,
                                          SBFileSpecRef platform_file);

LLDB_API SBFileSpecRef SBModuleGetRemoteInstallFileSpec(SBModuleRef instance);

LLDB_API bool SBModuleSetRemoteInstallFileSpec(SBModuleRef instance,
                                               SBFileSpecRef file);

LLDB_API ENUM(ByteOrder) SBModuleGetByteOrder(SBModuleRef instance);

LLDB_API uint32_t SBModuleGetAddressByteSize(SBModuleRef instance);

LLDB_API const char *SBModuleGetTriple(SBModuleRef instance);

LLDB_API const uint8_t *SBModuleGetUUIDBytes(SBModuleRef instance);

LLDB_API const char *SBModuleGetUUIDString(SBModuleRef instance);

LLDB_API SBSectionRef SBModuleFindSection(SBModuleRef instance,
                                          const char *sect_name);

LLDB_API SBAddressRef SBModuleResolveFileAddress(SBModuleRef instance,
                                                 lldb_addr_t vm_addr);

LLDB_API SBSymbolContextRef SBModuleResolveSymbolContextForAddress(
    SBModuleRef instance, SBAddressRef addr, uint32_t resolve_scope);

LLDB_API bool SBModuleGetDescription(SBModuleRef instance,
                                     SBStreamRef description);

LLDB_API uint32_t SBModuleGetNumCompileUnits(SBModuleRef instance);

LLDB_API SBCompileUnitRef SBModuleGetCompileUnitAtIndex(SBModuleRef instance,
                                                        uint32_t);

LLDB_API size_t SBModuleGetNumSymbols(SBModuleRef instance);

LLDB_API SBSymbolRef SBModuleGetSymbolAtIndex(SBModuleRef instance, size_t idx);

LLDB_API SBSymbolRef SBModuleFindSymbol(SBModuleRef instance, const char *name,
                                        ENUM(SymbolType) type);

LLDB_API SBSymbolContextListRef SBModuleFindSymbols(SBModuleRef instance,
                                                    const char *name,
                                                    ENUM(SymbolType) type);

LLDB_API size_t SBModuleGetNumSections(SBModuleRef instance);

LLDB_API SBSectionRef SBModuleGetSectionAtIndex(SBModuleRef instance,
                                                size_t idx);

LLDB_API SBSymbolContextListRef SBModuleFindFunctions(SBModuleRef instance,
                                                      const char *name,
                                                      uint32_t name_type_mask);

LLDB_API SBValueListRef SBModuleFindGlobalVariables(SBModuleRef instance,
                                                    SBTargetRef target,
                                                    const char *name,
                                                    uint32_t max_matches);

LLDB_API SBValueRef SBModuleFindFirstGlobalVariable(SBModuleRef instance,
                                                    SBTargetRef target,
                                                    const char *name);

LLDB_API SBTypeRef SBModuleFindFirstType(SBModuleRef instance,
                                         const char *name);

LLDB_API SBTypeListRef SBModuleFindTypes(SBModuleRef instance,
                                         const char *type);

LLDB_API SBTypeRef SBModuleGetTypeByID(SBModuleRef instance,
                                       lldb_user_id_t uid);

LLDB_API SBTypeRef SBModuleGetBasicType(SBModuleRef instance,
                                        ENUM(BasicType) type);

LLDB_API SBTypeListRef SBModuleGetTypes(SBModuleRef instance,
                                        uint32_t type_mask);

LLDB_API uint32_t SBModuleGetVersion(SBModuleRef instance, uint32_t *versions,
                                     uint32_t num_versions);

LLDB_API SBFileSpecRef SBModuleGetSymbolFileSpec(SBModuleRef instance);

LLDB_API SBAddressRef SBModuleGetObjectFileHeaderAddress(SBModuleRef instance);

LLDB_API SBAddressRef
SBModuleGetObjectFileEntryPointAddress(SBModuleRef instance);

LLDB_API uint32_t SBModuleGetNumberAllocatedModules();

LLDB_API void SBModuleGarbageCollectAllocatedModules();

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBModuleBinding_h_
