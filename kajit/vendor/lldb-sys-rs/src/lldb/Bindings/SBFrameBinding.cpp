//===-- SBFrameBinding.cpp --------------------------------------*- C++ -*-===//
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

SBFrameRef CreateSBFrame() {
  return reinterpret_cast<SBFrameRef>(new SBFrame());
}

SBFrameRef CloneSBFrame(SBFrameRef instance) {
  return reinterpret_cast<SBFrameRef>(
      new SBFrame(*reinterpret_cast<SBFrame *>(instance)));
}

void DisposeSBFrame(SBFrameRef instance) {
  delete reinterpret_cast<SBFrame *>(instance);
}

bool SBFrameIsEqual(SBFrameRef instance, SBFrameRef that) {
  SBFrame *unwrapped = reinterpret_cast<SBFrame *>(instance);
  return unwrapped->IsEqual(*reinterpret_cast<SBFrame *>(that));
}

bool SBFrameIsValid(SBFrameRef instance) {
  SBFrame *unwrapped = reinterpret_cast<SBFrame *>(instance);
  return unwrapped->IsValid();
}

uint32_t SBFrameGetFrameID(SBFrameRef instance) {
  SBFrame *unwrapped = reinterpret_cast<SBFrame *>(instance);
  return unwrapped->GetFrameID();
}

lldb_addr_t SBFrameGetCFA(SBFrameRef instance) {
  SBFrame *unwrapped = reinterpret_cast<SBFrame *>(instance);
  return unwrapped->GetCFA();
}

lldb_addr_t SBFrameGetPC(SBFrameRef instance) {
  SBFrame *unwrapped = reinterpret_cast<SBFrame *>(instance);
  return unwrapped->GetPC();
}

bool SBFrameSetPC(SBFrameRef instance, lldb_addr_t new_pc) {
  SBFrame *unwrapped = reinterpret_cast<SBFrame *>(instance);
  return unwrapped->SetPC(new_pc);
}

lldb_addr_t SBFrameGetSP(SBFrameRef instance) {
  SBFrame *unwrapped = reinterpret_cast<SBFrame *>(instance);
  return unwrapped->GetSP();
}

lldb_addr_t SBFrameGetFP(SBFrameRef instance) {
  SBFrame *unwrapped = reinterpret_cast<SBFrame *>(instance);
  return unwrapped->GetFP();
}

SBAddressRef SBFrameGetPCAddress(SBFrameRef instance) {
  SBFrame *unwrapped = reinterpret_cast<SBFrame *>(instance);
  return reinterpret_cast<SBAddressRef>(
      new SBAddress(unwrapped->GetPCAddress()));
}

SBSymbolContextRef SBFrameGetSymbolContext(SBFrameRef instance,
                                           uint32_t resolve_scope) {
  SBFrame *unwrapped = reinterpret_cast<SBFrame *>(instance);
  return reinterpret_cast<SBSymbolContextRef>(
      new SBSymbolContext(unwrapped->GetSymbolContext(resolve_scope)));
}

SBModuleRef SBFrameGetModule(SBFrameRef instance) {
  SBFrame *unwrapped = reinterpret_cast<SBFrame *>(instance);
  return reinterpret_cast<SBModuleRef>(new SBModule(unwrapped->GetModule()));
}

SBCompileUnitRef SBFrameGetCompileUnit(SBFrameRef instance) {
  SBFrame *unwrapped = reinterpret_cast<SBFrame *>(instance);
  return reinterpret_cast<SBCompileUnitRef>(
      new SBCompileUnit(unwrapped->GetCompileUnit()));
}

SBFunctionRef SBFrameGetFunction(SBFrameRef instance) {
  SBFrame *unwrapped = reinterpret_cast<SBFrame *>(instance);
  return reinterpret_cast<SBFunctionRef>(
      new SBFunction(unwrapped->GetFunction()));
}

SBSymbolRef SBFrameGetSymbol(SBFrameRef instance) {
  SBFrame *unwrapped = reinterpret_cast<SBFrame *>(instance);
  return reinterpret_cast<SBSymbolRef>(new SBSymbol(unwrapped->GetSymbol()));
}

SBBlockRef SBFrameGetBlock(SBFrameRef instance) {
  SBFrame *unwrapped = reinterpret_cast<SBFrame *>(instance);
  return reinterpret_cast<SBBlockRef>(new SBBlock(unwrapped->GetBlock()));
}

const char *SBFrameGetFunctionName(SBFrameRef instance) {
  SBFrame *unwrapped = reinterpret_cast<SBFrame *>(instance);
  return unwrapped->GetFunctionName();
}

const char *SBFrameGetDisplayFunctionName(SBFrameRef instance) {
  SBFrame *unwrapped = reinterpret_cast<SBFrame *>(instance);
  return unwrapped->GetDisplayFunctionName();
}

lldb::LanguageType SBFrameGuessLanguage(SBFrameRef instance) {
  SBFrame *unwrapped = reinterpret_cast<SBFrame *>(instance);
  return unwrapped->GuessLanguage();
}

bool SBFrameIsInlined(SBFrameRef instance) {
  SBFrame *unwrapped = reinterpret_cast<SBFrame *>(instance);
  return unwrapped->IsInlined();
}

bool SBFrameIsArtificial(SBFrameRef instance) {
  SBFrame *unwrapped = reinterpret_cast<SBFrame *>(instance);
  return unwrapped->IsArtificial();
}

SBValueRef SBFrameEvaluateExpression(SBFrameRef instance, const char *expr,
                                     SBExpressionOptionsRef options) {
  SBFrame *unwrapped = reinterpret_cast<SBFrame *>(instance);
  return reinterpret_cast<SBValueRef>(new SBValue(unwrapped->EvaluateExpression(
      expr, *reinterpret_cast<SBExpressionOptions *>(options))));
}

SBBlockRef SBFrameGetFrameBlock(SBFrameRef instance) {
  SBFrame *unwrapped = reinterpret_cast<SBFrame *>(instance);
  return reinterpret_cast<SBBlockRef>(new SBBlock(unwrapped->GetFrameBlock()));
}

SBLineEntryRef SBFrameGetLineEntry(SBFrameRef instance) {
  SBFrame *unwrapped = reinterpret_cast<SBFrame *>(instance);
  return reinterpret_cast<SBLineEntryRef>(
      new SBLineEntry(unwrapped->GetLineEntry()));
}

SBThreadRef SBFrameGetThread(SBFrameRef instance) {
  SBFrame *unwrapped = reinterpret_cast<SBFrame *>(instance);
  return reinterpret_cast<SBThreadRef>(new SBThread(unwrapped->GetThread()));
}

const char *SBFrameDisassemble(SBFrameRef instance) {
  SBFrame *unwrapped = reinterpret_cast<SBFrame *>(instance);
  return unwrapped->Disassemble();
}

void SBFrameClear(SBFrameRef instance) {
  SBFrame *unwrapped = reinterpret_cast<SBFrame *>(instance);
  unwrapped->Clear();
}

SBValueListRef SBFrameGetVariables(SBFrameRef instance,
                                   SBVariablesOptionsRef options) {
  SBFrame *unwrapped = reinterpret_cast<SBFrame *>(instance);
  return reinterpret_cast<SBValueListRef>(
      new SBValueList(unwrapped->GetVariables(
          *reinterpret_cast<SBVariablesOptions *>(options))));
}

SBValueListRef SBFrameGetRegisters(SBFrameRef instance) {
  SBFrame *unwrapped = reinterpret_cast<SBFrame *>(instance);
  return reinterpret_cast<SBValueListRef>(
      new SBValueList(unwrapped->GetRegisters()));
}

SBValueRef SBFrameFindRegister(SBFrameRef instance, const char *name) {
  SBFrame *unwrapped = reinterpret_cast<SBFrame *>(instance);
  return reinterpret_cast<SBValueRef>(
      new SBValue(unwrapped->FindRegister(name)));
}

SBValueRef SBFrameFindVariable(SBFrameRef instance, const char *var_name) {
  SBFrame *unwrapped = reinterpret_cast<SBFrame *>(instance);
  return reinterpret_cast<SBValueRef>(
      new SBValue(unwrapped->FindVariable(var_name)));
}

SBValueRef SBFrameFindVariable2(SBFrameRef instance, const char *var_name,
                                lldb::DynamicValueType use_dynamic) {
  SBFrame *unwrapped = reinterpret_cast<SBFrame *>(instance);
  return reinterpret_cast<SBValueRef>(
      new SBValue(unwrapped->FindVariable(var_name, use_dynamic)));
}

SBValueRef
SBFrameGetValueForVariablePath(SBFrameRef instance, const char *var_expr_cstr,
                               enum lldb::DynamicValueType use_dynamic) {
  SBFrame *unwrapped = reinterpret_cast<SBFrame *>(instance);
  return reinterpret_cast<SBValueRef>(new SBValue(
      unwrapped->GetValueForVariablePath(var_expr_cstr, use_dynamic)));
}

SBValueRef SBFrameGetValueForVariablePath2(SBFrameRef instance,
                                           const char *var_path) {
  SBFrame *unwrapped = reinterpret_cast<SBFrame *>(instance);
  return reinterpret_cast<SBValueRef>(
      new SBValue(unwrapped->GetValueForVariablePath(var_path)));
}

SBValueRef SBFrameFindValue(SBFrameRef instance, const char *name,
                            enum lldb::ValueType value_type) {
  SBFrame *unwrapped = reinterpret_cast<SBFrame *>(instance);
  return reinterpret_cast<SBValueRef>(
      new SBValue(unwrapped->FindValue(name, value_type)));
}

SBValueRef SBFrameFindValue2(SBFrameRef instance, const char *name,
                             enum lldb::ValueType value_type,
                             lldb::DynamicValueType use_dynamic) {
  SBFrame *unwrapped = reinterpret_cast<SBFrame *>(instance);
  return reinterpret_cast<SBValueRef>(
      new SBValue(unwrapped->FindValue(name, value_type, use_dynamic)));
}

bool SBFrameGetDescription(SBFrameRef instance, SBStreamRef description) {
  SBFrame *unwrapped = reinterpret_cast<SBFrame *>(instance);
  return unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description));
}

#ifdef __cplusplus
}
#endif
