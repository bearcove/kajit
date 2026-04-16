//===-- SBFrameBinding.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBFrameBinding_h_
#define LLDB_SBFrameBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBFrameRef CreateSBFrame(void);

LLDB_API SBFrameRef CloneSBFrame(SBFrameRef instance);

LLDB_API void DisposeSBFrame(SBFrameRef instance);

LLDB_API bool SBFrameIsEqual(SBFrameRef instance, SBFrameRef that);

LLDB_API bool SBFrameIsValid(SBFrameRef instance);

LLDB_API uint32_t SBFrameGetFrameID(SBFrameRef instance);

LLDB_API lldb_addr_t SBFrameGetCFA(SBFrameRef instance);

LLDB_API lldb_addr_t SBFrameGetPC(SBFrameRef instance);

LLDB_API bool SBFrameSetPC(SBFrameRef instance, lldb_addr_t new_pc);

LLDB_API lldb_addr_t SBFrameGetSP(SBFrameRef instance);

LLDB_API lldb_addr_t SBFrameGetFP(SBFrameRef instance);

LLDB_API SBAddressRef SBFrameGetPCAddress(SBFrameRef instance);

LLDB_API SBSymbolContextRef SBFrameGetSymbolContext(SBFrameRef instance,
                                                    uint32_t resolve_scope);

LLDB_API SBModuleRef SBFrameGetModule(SBFrameRef instance);

LLDB_API SBCompileUnitRef SBFrameGetCompileUnit(SBFrameRef instance);

LLDB_API SBFunctionRef SBFrameGetFunction(SBFrameRef instance);

LLDB_API SBSymbolRef SBFrameGetSymbol(SBFrameRef instance);

LLDB_API SBBlockRef SBFrameGetBlock(SBFrameRef instance);

LLDB_API const char *SBFrameGetFunctionName(SBFrameRef instance);

LLDB_API const char *SBFrameGetDisplayFunctionName(SBFrameRef instance);

LLDB_API ENUM(LanguageType) SBFrameGuessLanguage(SBFrameRef instance);

LLDB_API bool SBFrameIsInlined(SBFrameRef instance);

LLDB_API bool SBFrameIsArtificial(SBFrameRef instance);

LLDB_API SBValueRef SBFrameEvaluateExpression(SBFrameRef instance,
                                              const char *expr,
                                              SBExpressionOptionsRef options);

LLDB_API SBBlockRef SBFrameGetFrameBlock(SBFrameRef instance);

LLDB_API SBLineEntryRef SBFrameGetLineEntry(SBFrameRef instance);

LLDB_API SBThreadRef SBFrameGetThread(SBFrameRef instance);

LLDB_API const char *SBFrameDisassemble(SBFrameRef instance);

LLDB_API void SBFrameClear(SBFrameRef instance);

LLDB_API SBValueListRef SBFrameGetVariables(SBFrameRef instance,
                                            SBVariablesOptionsRef options);

LLDB_API SBValueListRef SBFrameGetRegisters(SBFrameRef instance);

LLDB_API SBValueRef SBFrameFindRegister(SBFrameRef instance, const char *name);

LLDB_API SBValueRef SBFrameFindVariable(SBFrameRef instance,
                                        const char *var_name);

LLDB_API SBValueRef SBFrameFindVariable2(SBFrameRef instance,
                                         const char *var_name,
                                         ENUM(DynamicValueType) use_dynamic);

LLDB_API SBValueRef SBFrameGetValueForVariablePath(SBFrameRef instance,
                                                   const char *var_expr_cstr,
                                                   ENUM(DynamicValueType)
                                                       use_dynamic);

LLDB_API SBValueRef SBFrameGetValueForVariablePath2(SBFrameRef instance,
                                                    const char *var_path);

LLDB_API SBValueRef SBFrameFindValue(SBFrameRef instance, const char *name,
                                     ENUM(ValueType) value_type);

LLDB_API SBValueRef SBFrameFindValue2(SBFrameRef instance, const char *name,
                                      ENUM(ValueType) value_type,
                                      ENUM(DynamicValueType) use_dynamic);

LLDB_API bool SBFrameGetDescription(SBFrameRef instance,
                                    SBStreamRef description);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBFrameBinding_h_
