//===-- SBModuleBinding.cpp -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/LLDB.h"
#include "lldb/Bindings/LLDBBinding.h"

using namespace lldb;

#ifdef __cplusplus
extern "C" {
#endif

SBModuleRef CreateSBModule() {
  return reinterpret_cast<SBModuleRef>(new SBModule());
}

SBModuleRef CreateSBModule2(SBModuleSpecRef module_spec) {
  return reinterpret_cast<SBModuleRef>(
      new SBModule(*reinterpret_cast<SBModuleSpec *>(module_spec)));
}

SBModuleRef CreateSBModule3(SBProcessRef process, lldb_addr_t header_addr) {
  return reinterpret_cast<SBModuleRef>(
      new SBModule(*reinterpret_cast<SBProcess *>(process), header_addr));
}

SBModuleRef CloneSBModule(SBModuleRef instance) {
  return reinterpret_cast<SBModuleRef>(
      new SBModule(*reinterpret_cast<SBModule *>(instance)));
}

void DisposeSBModule(SBModuleRef instance) {
  delete reinterpret_cast<SBModule *>(instance);
}

bool SBModuleIsValid(SBModuleRef instance) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  return unwrapped->IsValid();
}

void SBModuleClear(SBModuleRef instance) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  unwrapped->Clear();
}

bool SBModuleIsFileBacked(SBModuleRef instance) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  return unwrapped->IsFileBacked();
}

SBFileSpecRef SBModuleGetFileSpec(SBModuleRef instance) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  return reinterpret_cast<SBFileSpecRef>(
      new SBFileSpec(unwrapped->GetFileSpec()));
}

SBFileSpecRef SBModuleGetPlatformFileSpec(SBModuleRef instance) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  return reinterpret_cast<SBFileSpecRef>(
      new SBFileSpec(unwrapped->GetPlatformFileSpec()));
}

bool SBModuleSetPlatformFileSpec(SBModuleRef instance,
                                 SBFileSpecRef platform_file) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  return unwrapped->SetPlatformFileSpec(
      *reinterpret_cast<SBFileSpec *>(platform_file));
}

SBFileSpecRef SBModuleGetRemoteInstallFileSpec(SBModuleRef instance) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  return reinterpret_cast<SBFileSpecRef>(
      new SBFileSpec(unwrapped->GetRemoteInstallFileSpec()));
}

bool SBModuleSetRemoteInstallFileSpec(SBModuleRef instance,
                                      SBFileSpecRef file) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  return unwrapped->SetRemoteInstallFileSpec(
      *reinterpret_cast<SBFileSpec *>(file));
}

enum lldb::ByteOrder SBModuleGetByteOrder(SBModuleRef instance) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  return unwrapped->GetByteOrder();
}

uint32_t SBModuleGetAddressByteSize(SBModuleRef instance) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  return unwrapped->GetAddressByteSize();
}

const char *SBModuleGetTriple(SBModuleRef instance) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  return unwrapped->GetTriple();
}

const uint8_t *SBModuleGetUUIDBytes(SBModuleRef instance) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  return unwrapped->GetUUIDBytes();
}

const char *SBModuleGetUUIDString(SBModuleRef instance) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  return unwrapped->GetUUIDString();
}

SBSectionRef SBModuleFindSection(SBModuleRef instance, const char *sect_name) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  return reinterpret_cast<SBSectionRef>(
      new SBSection(unwrapped->FindSection(sect_name)));
}

SBAddressRef SBModuleResolveFileAddress(SBModuleRef instance,
                                        lldb_addr_t vm_addr) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  return reinterpret_cast<SBAddressRef>(
      new SBAddress(unwrapped->ResolveFileAddress(vm_addr)));
}

SBSymbolContextRef
SBModuleResolveSymbolContextForAddress(SBModuleRef instance, SBAddressRef addr,
                                       uint32_t resolve_scope) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  return reinterpret_cast<SBSymbolContextRef>(
      new SBSymbolContext(unwrapped->ResolveSymbolContextForAddress(
          *reinterpret_cast<SBAddress *>(addr), resolve_scope)));
}

bool SBModuleGetDescription(SBModuleRef instance, SBStreamRef description) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  return unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description));
}

uint32_t SBModuleGetNumCompileUnits(SBModuleRef instance) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  return unwrapped->GetNumCompileUnits();
}

SBCompileUnitRef SBModuleGetCompileUnitAtIndex(SBModuleRef instance,
                                               uint32_t var0) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  return reinterpret_cast<SBCompileUnitRef>(
      new SBCompileUnit(unwrapped->GetCompileUnitAtIndex(var0)));
}

size_t SBModuleGetNumSymbols(SBModuleRef instance) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  return unwrapped->GetNumSymbols();
}

SBSymbolRef SBModuleGetSymbolAtIndex(SBModuleRef instance, size_t idx) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  return reinterpret_cast<SBSymbolRef>(
      new SBSymbol(unwrapped->GetSymbolAtIndex(idx)));
}

SBSymbolRef SBModuleFindSymbol(SBModuleRef instance, const char *name,
                               lldb::SymbolType type) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  return reinterpret_cast<SBSymbolRef>(
      new SBSymbol(unwrapped->FindSymbol(name, type)));
}

SBSymbolContextListRef SBModuleFindSymbols(SBModuleRef instance,
                                           const char *name,
                                           lldb::SymbolType type) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  return reinterpret_cast<SBSymbolContextListRef>(
      new SBSymbolContextList(unwrapped->FindSymbols(name, type)));
}

size_t SBModuleGetNumSections(SBModuleRef instance) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  return unwrapped->GetNumSections();
}

SBSectionRef SBModuleGetSectionAtIndex(SBModuleRef instance, size_t idx) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  return reinterpret_cast<SBSectionRef>(
      new SBSection(unwrapped->GetSectionAtIndex(idx)));
}

SBSymbolContextListRef SBModuleFindFunctions(SBModuleRef instance,
                                             const char *name,
                                             uint32_t name_type_mask) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  return reinterpret_cast<SBSymbolContextListRef>(
      new SBSymbolContextList(unwrapped->FindFunctions(name, name_type_mask)));
}

SBValueListRef SBModuleFindGlobalVariables(SBModuleRef instance,
                                           SBTargetRef target, const char *name,
                                           uint32_t max_matches) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  return reinterpret_cast<SBValueListRef>(
      new SBValueList(unwrapped->FindGlobalVariables(
          *reinterpret_cast<SBTarget *>(target), name, max_matches)));
}

SBValueRef SBModuleFindFirstGlobalVariable(SBModuleRef instance,
                                           SBTargetRef target,
                                           const char *name) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  return reinterpret_cast<SBValueRef>(
      new SBValue(unwrapped->FindFirstGlobalVariable(
          *reinterpret_cast<SBTarget *>(target), name)));
}

SBTypeRef SBModuleFindFirstType(SBModuleRef instance, const char *name) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  return reinterpret_cast<SBTypeRef>(
      new SBType(unwrapped->FindFirstType(name)));
}

SBTypeListRef SBModuleFindTypes(SBModuleRef instance, const char *type) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  return reinterpret_cast<SBTypeListRef>(
      new SBTypeList(unwrapped->FindTypes(type)));
}

SBTypeRef SBModuleGetTypeByID(SBModuleRef instance, lldb_user_id_t uid) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  return reinterpret_cast<SBTypeRef>(new SBType(unwrapped->GetTypeByID(uid)));
}

SBTypeRef SBModuleGetBasicType(SBModuleRef instance, lldb::BasicType type) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  return reinterpret_cast<SBTypeRef>(new SBType(unwrapped->GetBasicType(type)));
}

SBTypeListRef SBModuleGetTypes(SBModuleRef instance, uint32_t type_mask) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  return reinterpret_cast<SBTypeListRef>(
      new SBTypeList(unwrapped->GetTypes(type_mask)));
}

uint32_t SBModuleGetVersion(SBModuleRef instance, uint32_t *versions,
                            uint32_t num_versions) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  return unwrapped->GetVersion(versions, num_versions);
}

SBFileSpecRef SBModuleGetSymbolFileSpec(SBModuleRef instance) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  return reinterpret_cast<SBFileSpecRef>(
      new SBFileSpec(unwrapped->GetSymbolFileSpec()));
}

SBAddressRef SBModuleGetObjectFileHeaderAddress(SBModuleRef instance) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  return reinterpret_cast<SBAddressRef>(
      new SBAddress(unwrapped->GetObjectFileHeaderAddress()));
}

SBAddressRef SBModuleGetObjectFileEntryPointAddress(SBModuleRef instance) {
  SBModule *unwrapped = reinterpret_cast<SBModule *>(instance);
  return reinterpret_cast<SBAddressRef>(
      new SBAddress(unwrapped->GetObjectFileEntryPointAddress()));
}

uint32_t SBModuleGetNumberAllocatedModules() {
  return SBModule::GetNumberAllocatedModules();
}

void SBModuleGarbageCollectAllocatedModules() {
  SBModule::GarbageCollectAllocatedModules();
}

#ifdef __cplusplus
}
#endif
