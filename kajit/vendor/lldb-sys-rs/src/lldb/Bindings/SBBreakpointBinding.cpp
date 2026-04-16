//===-- SBBreakpointBinding.cpp ---------------------------------*- C++ -*-===//
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

SBBreakpointRef CreateSBBreakpoint() {
  return reinterpret_cast<SBBreakpointRef>(new SBBreakpoint());
}

SBBreakpointRef CloneSBBreakpoint(SBBreakpointRef instance) {
  return reinterpret_cast<SBBreakpointRef>(
      new SBBreakpoint(*reinterpret_cast<SBBreakpoint *>(instance)));
}

void DisposeSBBreakpoint(SBBreakpointRef instance) {
  delete reinterpret_cast<SBBreakpoint *>(instance);
}

lldb_break_id_t SBBreakpointGetID(SBBreakpointRef instance) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  return unwrapped->GetID();
}

bool SBBreakpointIsValid(SBBreakpointRef instance) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  return unwrapped->IsValid();
}

void SBBreakpointClearAllBreakpointSites(SBBreakpointRef instance) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  unwrapped->ClearAllBreakpointSites();
}

SBTargetRef SBBreakpointGetTarget(SBBreakpointRef instance) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  return reinterpret_cast<SBTargetRef>(new SBTarget(unwrapped->GetTarget()));
}

SBBreakpointLocationRef
SBBreakpointFindLocationByAddress(SBBreakpointRef instance,
                                  lldb_addr_t vm_addr) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  return reinterpret_cast<SBBreakpointLocationRef>(
      new SBBreakpointLocation(unwrapped->FindLocationByAddress(vm_addr)));
}

lldb_break_id_t SBBreakpointFindLocationIDByAddress(SBBreakpointRef instance,
                                                    lldb_addr_t vm_addr) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  return unwrapped->FindLocationIDByAddress(vm_addr);
}

SBBreakpointLocationRef
SBBreakpointFindLocationByID(SBBreakpointRef instance,
                             lldb_break_id_t bp_loc_id) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  return reinterpret_cast<SBBreakpointLocationRef>(
      new SBBreakpointLocation(unwrapped->FindLocationByID(bp_loc_id)));
}

SBBreakpointLocationRef SBBreakpointGetLocationAtIndex(SBBreakpointRef instance,
                                                       uint32_t index) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  return reinterpret_cast<SBBreakpointLocationRef>(
      new SBBreakpointLocation(unwrapped->GetLocationAtIndex(index)));
}

void SBBreakpointSetEnabled(SBBreakpointRef instance, bool enable) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  unwrapped->SetEnabled(enable);
}

bool SBBreakpointIsEnabled(SBBreakpointRef instance) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  return unwrapped->IsEnabled();
}

void SBBreakpointSetOneShot(SBBreakpointRef instance, bool one_shot) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  unwrapped->SetOneShot(one_shot);
}

bool SBBreakpointIsOneShot(SBBreakpointRef instance) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  return unwrapped->IsOneShot();
}

bool SBBreakpointIsInternal(SBBreakpointRef instance) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  return unwrapped->IsInternal();
}

uint32_t SBBreakpointGetHitCount(SBBreakpointRef instance) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  return unwrapped->GetHitCount();
}

void SBBreakpointSetIgnoreCount(SBBreakpointRef instance, uint32_t count) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  unwrapped->SetIgnoreCount(count);
}

uint32_t SBBreakpointGetIgnoreCount(SBBreakpointRef instance) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  return unwrapped->GetIgnoreCount();
}

void SBBreakpointSetCondition(SBBreakpointRef instance, const char *condition) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  unwrapped->SetCondition(condition);
}

const char *SBBreakpointGetCondition(SBBreakpointRef instance) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  return unwrapped->GetCondition();
}

void SBBreakpointSetAutoContinue(SBBreakpointRef instance, bool auto_continue) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  unwrapped->SetAutoContinue(auto_continue);
}

bool SBBreakpointGetAutoContinue(SBBreakpointRef instance) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  return unwrapped->GetAutoContinue();
}

void SBBreakpointSetThreadID(SBBreakpointRef instance,
                             lldb_tid_t sb_thread_id) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  unwrapped->SetThreadID(sb_thread_id);
}

lldb_tid_t SBBreakpointGetThreadID(SBBreakpointRef instance) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  return unwrapped->GetThreadID();
}

void SBBreakpointSetThreadIndex(SBBreakpointRef instance, uint32_t index) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  unwrapped->SetThreadIndex(index);
}

uint32_t SBBreakpointGetThreadIndex(SBBreakpointRef instance) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  return unwrapped->GetThreadIndex();
}

void SBBreakpointSetThreadName(SBBreakpointRef instance,
                               const char *thread_name) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  unwrapped->SetThreadName(thread_name);
}

const char *SBBreakpointGetThreadName(SBBreakpointRef instance) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  return unwrapped->GetThreadName();
}

void SBBreakpointSetQueueName(SBBreakpointRef instance,
                              const char *queue_name) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  unwrapped->SetQueueName(queue_name);
}

const char *SBBreakpointGetQueueName(SBBreakpointRef instance) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  return unwrapped->GetQueueName();
}

void SBBreakpointSetCallback(SBBreakpointRef instance,
                             lldb::SBBreakpointHitCallback callback,
                             void *baton) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  unwrapped->SetCallback(callback, baton);
}

SBErrorRef
SBBreakpointSetScriptCallbackFunction(SBBreakpointRef instance,
                                      const char *callback_function_name,
                                      SBStructuredDataRef extra_args) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  return reinterpret_cast<SBErrorRef>(
      new SBError(unwrapped->SetScriptCallbackFunction(
          callback_function_name,
          *reinterpret_cast<SBStructuredData *>(extra_args))));
}

void SBBreakpointSetCommandLineCommands(SBBreakpointRef instance,
                                        SBStringListRef commands) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  unwrapped->SetCommandLineCommands(
      *reinterpret_cast<SBStringList *>(commands));
}

bool SBBreakpointGetCommandLineCommands(SBBreakpointRef instance,
                                        SBStringListRef commands) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  return unwrapped->GetCommandLineCommands(
      *reinterpret_cast<SBStringList *>(commands));
}

SBErrorRef SBBreakpointSetScriptCallbackBody(SBBreakpointRef instance,
                                             const char *script_body_text) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  return reinterpret_cast<SBErrorRef>(
      new SBError(unwrapped->SetScriptCallbackBody(script_body_text)));
}

bool SBBreakpointAddName(SBBreakpointRef instance, const char *new_name) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  return unwrapped->AddName(new_name);
}

SBErrorRef SBBreakpointAddNameWithErrorHandling(SBBreakpointRef instance,
                                                const char *new_name) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  return reinterpret_cast<SBErrorRef>(
      new SBError(unwrapped->AddNameWithErrorHandling(new_name)));
}

void SBBreakpointRemoveName(SBBreakpointRef instance,
                            const char *name_to_remove) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  unwrapped->RemoveName(name_to_remove);
}

bool SBBreakpointMatchesName(SBBreakpointRef instance, const char *name) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  return unwrapped->MatchesName(name);
}

void SBBreakpointGetNames(SBBreakpointRef instance, SBStringListRef names) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  unwrapped->GetNames(*reinterpret_cast<SBStringList *>(names));
}

size_t SBBreakpointGetNumResolvedLocations(SBBreakpointRef instance) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  return unwrapped->GetNumResolvedLocations();
}

size_t SBBreakpointGetNumLocations(SBBreakpointRef instance) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  return unwrapped->GetNumLocations();
}

bool SBBreakpointGetDescription(SBBreakpointRef instance,
                                SBStreamRef description) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  return unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description));
}

bool SBBreakpointGetDescription2(SBBreakpointRef instance,
                                 SBStreamRef description,
                                 bool include_locations) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  return unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description),
                                   include_locations);
}

bool SBBreakpointEventIsBreakpointEvent(SBEventRef event) {
  return lldb::SBBreakpoint::EventIsBreakpointEvent(
      *reinterpret_cast<SBEvent *>(event));
}

lldb::BreakpointEventType
SBBreakpointGetBreakpointEventTypeFromEvent(SBEventRef event) {
  return lldb::SBBreakpoint::GetBreakpointEventTypeFromEvent(
      *reinterpret_cast<SBEvent *>(event));
}

SBBreakpointRef SBBreakpointGetBreakpointFromEvent(SBEventRef event) {
  return reinterpret_cast<SBBreakpointRef>(
      new SBBreakpoint(lldb::SBBreakpoint::GetBreakpointFromEvent(
          *reinterpret_cast<SBEvent *>(event))));
}

SBBreakpointLocationRef
SBBreakpointGetBreakpointLocationAtIndexFromEvent(SBEventRef event,
                                                  uint32_t loc_idx) {
  return reinterpret_cast<SBBreakpointLocationRef>(new SBBreakpointLocation(
      lldb::SBBreakpoint::GetBreakpointLocationAtIndexFromEvent(
          *reinterpret_cast<SBEvent *>(event), loc_idx)));
}

uint32_t SBBreakpointGetNumBreakpointLocationsFromEvent(SBEventRef event_sp) {
  return lldb::SBBreakpoint::GetNumBreakpointLocationsFromEvent(
      *reinterpret_cast<SBEvent *>(event_sp));
}

bool SBBreakpointIsHardware(SBBreakpointRef instance) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  return unwrapped->IsHardware();
}

SBErrorRef SBBreakpointAddLocation(SBBreakpointRef instance,
                                   SBAddressRef address) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  return reinterpret_cast<SBErrorRef>(new SBError(
      unwrapped->AddLocation(*reinterpret_cast<SBAddress *>(address))));
}

SBStructuredDataRef
SBBreakpointSerializeToStructuredData(SBBreakpointRef instance) {
  SBBreakpoint *unwrapped = reinterpret_cast<SBBreakpoint *>(instance);
  return reinterpret_cast<SBStructuredDataRef>(
      new SBStructuredData(unwrapped->SerializeToStructuredData()));
}

#ifdef __cplusplus
}
#endif
