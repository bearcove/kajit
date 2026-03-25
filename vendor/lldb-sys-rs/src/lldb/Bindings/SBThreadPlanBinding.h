//===-- SBThreadPlanBinding.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBThreadPlanBinding_h_
#define LLDB_SBThreadPlanBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBThreadPlanRef CreateSBThreadPlan(void);

LLDB_API SBThreadPlanRef CreateSBThreadPlan2(SBThreadRef thread,
                                             const char *class_name,
                                             SBStructuredDataRef args_data);

LLDB_API SBThreadPlanRef CloneSBThreadPlan(SBThreadPlanRef instance);

LLDB_API void DisposeSBThreadPlan(SBThreadPlanRef instance);

LLDB_API bool SBThreadPlanIsValid(SBThreadPlanRef instance);

LLDB_API void SBThreadPlanClear(SBThreadPlanRef instance);

LLDB_API ENUM(StopReason) SBThreadPlanGetStopReason(SBThreadPlanRef instance);

LLDB_API size_t SBThreadPlanGetStopReasonDataCount(SBThreadPlanRef instance);

LLDB_API uint64_t SBThreadPlanGetStopReasonDataAtIndex(SBThreadPlanRef instance,
                                                       uint32_t idx);

LLDB_API SBThreadRef SBThreadPlanGetThread(SBThreadPlanRef instance);

LLDB_API bool SBThreadPlanGetDescription(SBThreadPlanRef instance,
                                         SBStreamRef description);

LLDB_API void SBThreadPlanSetPlanComplete(SBThreadPlanRef instance,
                                          bool success);

LLDB_API bool SBThreadPlanIsPlanComplete(SBThreadPlanRef instance);

LLDB_API bool SBThreadPlanIsPlanStale(SBThreadPlanRef instance);

LLDB_API bool SBThreadPlanIsValid(SBThreadPlanRef instance);

LLDB_API bool SBThreadPlanGetStopOthers(SBThreadPlanRef instance);

LLDB_API void SBThreadPlanSetStopOthers(SBThreadPlanRef instance,
                                        bool stop_others);

LLDB_API SBThreadPlanRef SBThreadPlanQueueThreadPlanForStepOverRange(
    SBThreadPlanRef instance, SBAddressRef start_address,
    lldb_addr_t range_size, SBErrorRef error);

LLDB_API SBThreadPlanRef SBThreadPlanQueueThreadPlanForStepInRange(
    SBThreadPlanRef instance, SBAddressRef start_address,
    lldb_addr_t range_size, SBErrorRef error);

LLDB_API SBThreadPlanRef SBThreadPlanQueueThreadPlanForStepOut(
    SBThreadPlanRef instance, uint32_t frame_idx_to_step_to, bool first_insn,
    SBErrorRef error);

LLDB_API SBThreadPlanRef SBThreadPlanQueueThreadPlanForRunToAddress(
    SBThreadPlanRef instance, SBAddressRef address, SBErrorRef error);

LLDB_API SBThreadPlanRef SBThreadPlanQueueThreadPlanForStepScripted(
    SBThreadPlanRef instance, const char *script_class_name,
    SBStructuredDataRef args_data, SBErrorRef error);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBThreadPlanBinding_h_
