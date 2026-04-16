//===-- SBAddressBinding.cpp ------------------------------------*- C++ -*-===//
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

SBAddressRef CreateSBAddress() {
  return reinterpret_cast<SBAddressRef>(new SBAddress());
}

SBAddressRef CreateSBAddress2(SBSectionRef section, lldb_addr_t offset) {
  return reinterpret_cast<SBAddressRef>(
      new SBAddress(*reinterpret_cast<SBSection *>(section), offset));
}

SBAddressRef CreateSBAddress3(lldb_addr_t load_addr, SBTargetRef target) {
  return reinterpret_cast<SBAddressRef>(
      new SBAddress(load_addr, *reinterpret_cast<SBTarget *>(target)));
}

SBAddressRef CloneSBAddress(SBAddressRef instance) {
  return reinterpret_cast<SBAddressRef>(
      new SBAddress(*reinterpret_cast<SBAddress *>(instance)));
}

void DisposeSBAddress(SBAddressRef instance) {
  delete reinterpret_cast<SBAddress *>(instance);
}

bool SBAddressIsValid(SBAddressRef instance) {
  SBAddress *unwrapped = reinterpret_cast<SBAddress *>(instance);
  return unwrapped->IsValid();
}

void SBAddressClear(SBAddressRef instance) {
  SBAddress *unwrapped = reinterpret_cast<SBAddress *>(instance);
  unwrapped->Clear();
}

lldb_addr_t SBAddressGetFileAddress(SBAddressRef instance) {
  SBAddress *unwrapped = reinterpret_cast<SBAddress *>(instance);
  return unwrapped->GetFileAddress();
}

lldb_addr_t SBAddressGetLoadAddress(SBAddressRef instance, SBTargetRef target) {
  SBAddress *unwrapped = reinterpret_cast<SBAddress *>(instance);
  return unwrapped->GetLoadAddress(*reinterpret_cast<SBTarget *>(target));
}

void SBAddressSetAddress(SBAddressRef instance, SBSectionRef section,
                         lldb_addr_t offset) {
  SBAddress *unwrapped = reinterpret_cast<SBAddress *>(instance);
  unwrapped->SetAddress(*reinterpret_cast<SBSection *>(section), offset);
}

void SBAddressSetLoadAddress(SBAddressRef instance, lldb_addr_t load_addr,
                             SBTargetRef target) {
  SBAddress *unwrapped = reinterpret_cast<SBAddress *>(instance);
  unwrapped->SetLoadAddress(load_addr, *reinterpret_cast<SBTarget *>(target));
}

bool SBAddressOffsetAddress(SBAddressRef instance, lldb_addr_t offset) {
  SBAddress *unwrapped = reinterpret_cast<SBAddress *>(instance);
  return unwrapped->OffsetAddress(offset);
}

bool SBAddressGetDescription(SBAddressRef instance, SBStreamRef description) {
  SBAddress *unwrapped = reinterpret_cast<SBAddress *>(instance);
  return unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description));
}

SBSymbolContextRef SBAddressGetSymbolContext(SBAddressRef instance,
                                             uint32_t resolve_scope) {
  SBAddress *unwrapped = reinterpret_cast<SBAddress *>(instance);
  return reinterpret_cast<SBSymbolContextRef>(
      new SBSymbolContext(unwrapped->GetSymbolContext(resolve_scope)));
}

SBSectionRef SBAddressGetSection(SBAddressRef instance) {
  SBAddress *unwrapped = reinterpret_cast<SBAddress *>(instance);
  return reinterpret_cast<SBSectionRef>(new SBSection(unwrapped->GetSection()));
}

lldb_addr_t SBAddressGetOffset(SBAddressRef instance) {
  SBAddress *unwrapped = reinterpret_cast<SBAddress *>(instance);
  return unwrapped->GetOffset();
}

SBModuleRef SBAddressGetModule(SBAddressRef instance) {
  SBAddress *unwrapped = reinterpret_cast<SBAddress *>(instance);
  return reinterpret_cast<SBModuleRef>(new SBModule(unwrapped->GetModule()));
}

SBCompileUnitRef SBAddressGetCompileUnit(SBAddressRef instance) {
  SBAddress *unwrapped = reinterpret_cast<SBAddress *>(instance);
  return reinterpret_cast<SBCompileUnitRef>(
      new SBCompileUnit(unwrapped->GetCompileUnit()));
}

SBFunctionRef SBAddressGetFunction(SBAddressRef instance) {
  SBAddress *unwrapped = reinterpret_cast<SBAddress *>(instance);
  return reinterpret_cast<SBFunctionRef>(
      new SBFunction(unwrapped->GetFunction()));
}

SBBlockRef SBAddressGetBlock(SBAddressRef instance) {
  SBAddress *unwrapped = reinterpret_cast<SBAddress *>(instance);
  return reinterpret_cast<SBBlockRef>(new SBBlock(unwrapped->GetBlock()));
}

SBSymbolRef SBAddressGetSymbol(SBAddressRef instance) {
  SBAddress *unwrapped = reinterpret_cast<SBAddress *>(instance);
  return reinterpret_cast<SBSymbolRef>(new SBSymbol(unwrapped->GetSymbol()));
}

SBLineEntryRef SBAddressGetLineEntry(SBAddressRef instance) {
  SBAddress *unwrapped = reinterpret_cast<SBAddress *>(instance);
  return reinterpret_cast<SBLineEntryRef>(
      new SBLineEntry(unwrapped->GetLineEntry()));
}

bool SBAddressIsEqual(SBAddressRef instance, SBAddressRef other) {
  SBAddress *instance_unwrapped = reinterpret_cast<SBAddress *>(instance);
  SBAddress *other_unwrapped = reinterpret_cast<SBAddress *>(other);
  return *instance_unwrapped == *other_unwrapped;
}

#ifdef __cplusplus
}
#endif
