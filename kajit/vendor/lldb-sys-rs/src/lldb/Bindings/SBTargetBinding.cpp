//===-- SBTargetBinding.cpp -------------------------------------*- C++ -*-===//
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

SBTargetRef CreateSBTarget() {
  return reinterpret_cast<SBTargetRef>(new SBTarget());
}

SBTargetRef CloneSBTarget(SBTargetRef instance) {
  return reinterpret_cast<SBTargetRef>(
      new SBTarget(*reinterpret_cast<SBTarget *>(instance)));
}

void DisposeSBTarget(SBTargetRef instance) {
  delete reinterpret_cast<SBTarget *>(instance);
}

bool SBTargetIsValid(SBTargetRef instance) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return unwrapped->IsValid();
}

bool SBTargetEventIsTargetEvent(SBEventRef event) {
  return lldb::SBTarget::EventIsTargetEvent(
      *reinterpret_cast<SBEvent *>(event));
}

SBTargetRef SBTargetGetTargetFromEvent(SBEventRef event) {
  return reinterpret_cast<SBTargetRef>(new SBTarget(
      lldb::SBTarget::GetTargetFromEvent(*reinterpret_cast<SBEvent *>(event))));
}

uint32_t SBTargetGetNumModulesFromEvent(SBEventRef event) {
  return lldb::SBTarget::GetNumModulesFromEvent(
      *reinterpret_cast<SBEvent *>(event));
}

SBModuleRef SBTargetGetModuleAtIndexFromEvent(const uint32_t idx,
                                              SBEventRef event) {
  return reinterpret_cast<SBModuleRef>(
      new SBModule(lldb::SBTarget::GetModuleAtIndexFromEvent(
          idx, *reinterpret_cast<SBEvent *>(event))));
}

const char *SBTargetGetBroadcasterClassName() {
  return lldb::SBTarget::GetBroadcasterClassName();
}

SBProcessRef SBTargetGetProcess(SBTargetRef instance) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBProcessRef>(new SBProcess(unwrapped->GetProcess()));
}

void SBTargetSetCollectingStats(SBTargetRef instance, bool v) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  unwrapped->SetCollectingStats(v);
}

bool SBTargetGetCollectingStats(SBTargetRef instance) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return unwrapped->GetCollectingStats();
}

SBStructuredDataRef SBBTargetGetStatistics(SBTargetRef instance) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBStructuredDataRef>(
      new SBStructuredData(unwrapped->GetStatistics()));
}

SBPlatformRef SBTargetGetPlatform(SBTargetRef instance) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBPlatformRef>(
      new SBPlatform(unwrapped->GetPlatform()));
}

SBEnvironmentRef SBTargetGetEnvironment(SBTargetRef instance) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBEnvironmentRef>(
      new SBEnvironment(unwrapped->GetEnvironment()));
}

SBErrorRef SBTargetInstall(SBTargetRef instance) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBErrorRef>(new SBError(unwrapped->Install()));
}

SBProcessRef SBTargetLaunch(SBTargetRef instance, SBListenerRef listener,
                            const char **argv, const char **envp,
                            const char *stdin_path, const char *stdout_path,
                            const char *stderr_path,
                            const char *working_directory,
                            uint32_t launch_flags, bool stop_at_entry,
                            SBErrorRef error) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBProcessRef>(new SBProcess(unwrapped->Launch(
      *reinterpret_cast<SBListener *>(listener), argv, envp, stdin_path,
      stdout_path, stderr_path, working_directory, launch_flags, stop_at_entry,
      *reinterpret_cast<SBError *>(error))));
}

SBProcessRef SBTargetLaunchSimple(SBTargetRef instance, const char **argv,
                                  const char **envp,
                                  const char *working_directory) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBProcessRef>(
      new SBProcess(unwrapped->LaunchSimple(argv, envp, working_directory)));
}

SBProcessRef SBTargetLaunch2(SBTargetRef instance, SBLaunchInfoRef launch_info,
                             SBErrorRef error) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBProcessRef>(new SBProcess(
      unwrapped->Launch(*reinterpret_cast<SBLaunchInfo *>(launch_info),
                        *reinterpret_cast<SBError *>(error))));
}

SBProcessRef SBTargetLoadCore(SBTargetRef instance, const char *core_file,
                              SBErrorRef error) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBProcessRef>(new SBProcess(
      unwrapped->LoadCore(core_file, *reinterpret_cast<SBError *>(error))));
}

SBProcessRef SBTargetAttach(SBTargetRef instance, SBAttachInfoRef attach_info,
                            SBErrorRef error) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBProcessRef>(new SBProcess(
      unwrapped->Attach(*reinterpret_cast<SBAttachInfo *>(attach_info),
                        *reinterpret_cast<SBError *>(error))));
}

SBProcessRef SBTargetAttachToProcessWithID(SBTargetRef instance,
                                           SBListenerRef listener,
                                           lldb_pid_t pid, SBErrorRef error) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBProcessRef>(
      new SBProcess(unwrapped->AttachToProcessWithID(
          *reinterpret_cast<SBListener *>(listener), pid,
          *reinterpret_cast<SBError *>(error))));
}

SBProcessRef SBTargetAttachToProcessWithName(SBTargetRef instance,
                                             SBListenerRef listener,
                                             const char *name, bool wait_for,
                                             SBErrorRef error) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBProcessRef>(
      new SBProcess(unwrapped->AttachToProcessWithName(
          *reinterpret_cast<SBListener *>(listener), name, wait_for,
          *reinterpret_cast<SBError *>(error))));
}

SBProcessRef SBTargetConnectRemote(SBTargetRef instance, SBListenerRef listener,
                                   const char *url, const char *plugin_name,
                                   SBErrorRef error) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBProcessRef>(new SBProcess(unwrapped->ConnectRemote(
      *reinterpret_cast<SBListener *>(listener), url, plugin_name,
      *reinterpret_cast<SBError *>(error))));
}

SBFileSpecRef SBTargetGetExecutable(SBTargetRef instance) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBFileSpecRef>(
      new SBFileSpec(unwrapped->GetExecutable()));
}

void SBTargetAppendImageSearchPath(SBTargetRef instance, const char *from,
                                   const char *to, SBErrorRef error) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  unwrapped->AppendImageSearchPath(from, to,
                                   *reinterpret_cast<SBError *>(error));
}

bool SBTargetAddModule(SBTargetRef instance, SBModuleRef module) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return unwrapped->AddModule(*reinterpret_cast<SBModule *>(module));
}

SBModuleRef SBTargetAddModuleSpec(SBTargetRef instance,
                                  SBModuleSpecRef module_spec) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBModuleRef>(new SBModule(
      unwrapped->AddModule(*reinterpret_cast<SBModuleSpec *>(module_spec))));
}

uint32_t SBTargetGetNumModules(SBTargetRef instance) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return unwrapped->GetNumModules();
}

SBModuleRef SBTargetGetModuleAtIndex(SBTargetRef instance, uint32_t idx) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBModuleRef>(
      new SBModule(unwrapped->GetModuleAtIndex(idx)));
}

bool SBTargetRemoveModule(SBTargetRef instance, SBModuleRef module) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return unwrapped->RemoveModule(*reinterpret_cast<SBModule *>(module));
}

SBDebuggerRef SBTargetGetDebugger(SBTargetRef instance) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBDebuggerRef>(
      new SBDebugger(unwrapped->GetDebugger()));
}

SBModuleRef SBTargetFindModule(SBTargetRef instance, SBFileSpecRef file_spec) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBModuleRef>(new SBModule(
      unwrapped->FindModule(*reinterpret_cast<SBFileSpec *>(file_spec))));
}

SBSymbolContextListRef SBTargetFindCompileUnits(SBTargetRef instance,
                                                SBFileSpecRef file_spec) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBSymbolContextListRef>(new SBSymbolContextList(
      unwrapped->FindCompileUnits(*reinterpret_cast<SBFileSpec *>(file_spec))));
}

enum lldb::ByteOrder SBTargetGetByteOrder(SBTargetRef instance) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return unwrapped->GetByteOrder();
}

uint32_t SBTargetGetAddressByteSize(SBTargetRef instance) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return unwrapped->GetAddressByteSize();
}

const char *SBTargetGetTriple(SBTargetRef instance) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return unwrapped->GetTriple();
}

const char *SBTargetGetABIName(SBTargetRef instance) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return unwrapped->GetABIName();
}

uint32_t SBTargetGetDataByteSize(SBTargetRef instance) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return unwrapped->GetDataByteSize();
}

uint32_t SBTargetGetCodeByteSize(SBTargetRef instance) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return unwrapped->GetCodeByteSize();
}

uint32_t SBTargetGetMaximumNumberOfChildrenToDisplay(SBTargetRef instance) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return unwrapped->GetMaximumNumberOfChildrenToDisplay();
}

SBErrorRef SBTargetSetSectionLoadAddress(SBTargetRef instance,
                                         SBSectionRef section,
                                         lldb_addr_t section_base_addr) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBErrorRef>(
      new SBError(unwrapped->SetSectionLoadAddress(
          *reinterpret_cast<SBSection *>(section), section_base_addr)));
}

SBErrorRef SBTargetClearSectionLoadAddress(SBTargetRef instance,
                                           SBSectionRef section) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBErrorRef>(
      new SBError(unwrapped->ClearSectionLoadAddress(
          *reinterpret_cast<SBSection *>(section))));
}

SBErrorRef SBTargetSetModuleLoadAddress(SBTargetRef instance,
                                        SBModuleRef module,
                                        int64_t sections_offset) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBErrorRef>(
      new SBError(unwrapped->SetModuleLoadAddress(
          *reinterpret_cast<SBModule *>(module), sections_offset)));
}

SBErrorRef SBTargetClearModuleLoadAddress(SBTargetRef instance,
                                          SBModuleRef module) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBErrorRef>(
      new SBError(unwrapped->ClearModuleLoadAddress(
          *reinterpret_cast<SBModule *>(module))));
}

SBSymbolContextListRef SBTargetFindFunctions(SBTargetRef instance,
                                             const char *name,
                                             uint32_t name_type_mask) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBSymbolContextListRef>(
      new SBSymbolContextList(unwrapped->FindFunctions(name, name_type_mask)));
}

SBValueListRef SBTargetFindGlobalVariables(SBTargetRef instance,
                                           const char *name,
                                           uint32_t max_matches) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBValueListRef>(
      new SBValueList(unwrapped->FindGlobalVariables(name, max_matches)));
}

SBValueRef SBTargetFindFirstGlobalVariable(SBTargetRef instance,
                                           const char *name) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBValueRef>(
      new SBValue(unwrapped->FindFirstGlobalVariable(name)));
}

SBValueListRef SBTargetFindGlobalVariables2(SBTargetRef instance,
                                            const char *name,
                                            uint32_t max_matches,
                                            enum lldb::MatchType matchtype) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBValueListRef>(new SBValueList(
      unwrapped->FindGlobalVariables(name, max_matches, matchtype)));
}

SBSymbolContextListRef
SBTargetFindGlobalFunctions(SBTargetRef instance, const char *name,
                            uint32_t max_matches,
                            enum lldb::MatchType matchtype) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBSymbolContextListRef>(new SBSymbolContextList(
      unwrapped->FindGlobalFunctions(name, max_matches, matchtype)));
}

void SBTargetClear(SBTargetRef instance) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  unwrapped->Clear();
}

SBAddressRef SBTargetResolveFileAddress(SBTargetRef instance,
                                        lldb_addr_t file_addr) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBAddressRef>(
      new SBAddress(unwrapped->ResolveFileAddress(file_addr)));
}

SBAddressRef SBTargetResolveLoadAddress(SBTargetRef instance,
                                        lldb_addr_t vm_addr) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBAddressRef>(
      new SBAddress(unwrapped->ResolveLoadAddress(vm_addr)));
}

SBAddressRef SBTargetResolvePastLoadAddress(SBTargetRef instance,
                                            uint32_t stop_id,
                                            lldb_addr_t vm_addr) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBAddressRef>(
      new SBAddress(unwrapped->ResolvePastLoadAddress(stop_id, vm_addr)));
}

SBSymbolContextRef
SBTargetResolveSymbolContextForAddress(SBTargetRef instance, SBAddressRef addr,
                                       uint32_t resolve_scope) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBSymbolContextRef>(
      new SBSymbolContext(unwrapped->ResolveSymbolContextForAddress(
          *reinterpret_cast<SBAddress *>(addr), resolve_scope)));
}

size_t SBTargetReadMemory(SBTargetRef instance, SBAddressRef addr, void *buf,
                          size_t size, SBErrorRef error) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return unwrapped->ReadMemory(*reinterpret_cast<SBAddress *>(addr), buf, size,
                               *reinterpret_cast<SBError *>(error));
}

SBBreakpointRef SBTargetBreakpointCreateByLocation(SBTargetRef instance,
                                                   const char *file,
                                                   uint32_t line) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBBreakpointRef>(
      new SBBreakpoint(unwrapped->BreakpointCreateByLocation(file, line)));
}

SBBreakpointRef SBTargetBreakpointCreateByLocation2(SBTargetRef instance,
                                                    SBFileSpecRef file_spec,
                                                    uint32_t line) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBBreakpointRef>(
      new SBBreakpoint(unwrapped->BreakpointCreateByLocation(
          *reinterpret_cast<SBFileSpec *>(file_spec), line)));
}

SBBreakpointRef SBTargetBreakpointCreateByLocation3(SBTargetRef instance,
                                                    SBFileSpecRef file_spec,
                                                    uint32_t line,
                                                    lldb_addr_t offset) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBBreakpointRef>(
      new SBBreakpoint(unwrapped->BreakpointCreateByLocation(
          *reinterpret_cast<SBFileSpec *>(file_spec), line, offset)));
}

SBBreakpointRef SBTargetBreakpointCreateByLocation4(
    SBTargetRef instance, SBFileSpecRef file_spec, uint32_t line,
    lldb_addr_t offset, SBFileSpecListRef module_list) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBBreakpointRef>(
      new SBBreakpoint(unwrapped->BreakpointCreateByLocation(
          *reinterpret_cast<SBFileSpec *>(file_spec), line, offset,
          *reinterpret_cast<SBFileSpecList *>(module_list))));
}

SBBreakpointRef SBTargetBreakpointCreateByLocation5(
    SBTargetRef instance, SBFileSpecRef file_spec, uint32_t line,
    uint32_t column, lldb_addr_t offset, SBFileSpecListRef module_list) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBBreakpointRef>(
      new SBBreakpoint(unwrapped->BreakpointCreateByLocation(
          *reinterpret_cast<SBFileSpec *>(file_spec), line, column, offset,
          *reinterpret_cast<SBFileSpecList *>(module_list))));
}

SBBreakpointRef SBTargetBreakpointCreateByLocation6(
    SBTargetRef instance, SBFileSpecRef file_spec, uint32_t line,
    uint32_t column, lldb_addr_t offset, SBFileSpecListRef module_list,
    bool move_to_nearest_code) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBBreakpointRef>(
      new SBBreakpoint(unwrapped->BreakpointCreateByLocation(
          *reinterpret_cast<SBFileSpec *>(file_spec), line, column, offset,
          *reinterpret_cast<SBFileSpecList *>(module_list),
          move_to_nearest_code)));
}

SBBreakpointRef SBTargetBreakpointCreateByName(SBTargetRef instance,
                                               const char *symbol_name,
                                               const char *module_name) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBBreakpointRef>(new SBBreakpoint(
      unwrapped->BreakpointCreateByName(symbol_name, module_name)));
}

SBBreakpointRef
SBTargetBreakpointCreateByName2(SBTargetRef instance, const char *symbol_name,
                                SBFileSpecListRef module_list,
                                SBFileSpecListRef comp_unit_list) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBBreakpointRef>(
      new SBBreakpoint(unwrapped->BreakpointCreateByName(
          symbol_name, *reinterpret_cast<SBFileSpecList *>(module_list),
          *reinterpret_cast<SBFileSpecList *>(comp_unit_list))));
}

SBBreakpointRef SBTargetBreakpointCreateByName3(
    SBTargetRef instance, const char *symbol_name, uint32_t name_type_mask,
    SBFileSpecListRef module_list, SBFileSpecListRef comp_unit_list) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBBreakpointRef>(
      new SBBreakpoint(unwrapped->BreakpointCreateByName(
          symbol_name, name_type_mask,
          *reinterpret_cast<SBFileSpecList *>(module_list),
          *reinterpret_cast<SBFileSpecList *>(comp_unit_list))));
}

SBBreakpointRef
SBTargetBreakpointCreateByNames(SBTargetRef instance, const char **symbol_name,
                                uint32_t num_names, uint32_t name_type_mask,
                                SBFileSpecListRef module_list,
                                SBFileSpecListRef comp_unit_list) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBBreakpointRef>(
      new SBBreakpoint(unwrapped->BreakpointCreateByNames(
          symbol_name, num_names, name_type_mask,
          *reinterpret_cast<SBFileSpecList *>(module_list),
          *reinterpret_cast<SBFileSpecList *>(comp_unit_list))));
}

SBBreakpointRef SBTargetBreakpointCreateByNames2(
    SBTargetRef instance, const char **symbol_name, uint32_t num_names,
    uint32_t name_type_mask, lldb::LanguageType symbol_language,
    SBFileSpecListRef module_list, SBFileSpecListRef comp_unit_list) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBBreakpointRef>(
      new SBBreakpoint(unwrapped->BreakpointCreateByNames(
          symbol_name, num_names, name_type_mask, symbol_language,
          *reinterpret_cast<SBFileSpecList *>(module_list),
          *reinterpret_cast<SBFileSpecList *>(comp_unit_list))));
}

SBBreakpointRef SBTargetBreakpointCreateByNames3(
    SBTargetRef instance, const char **symbol_name, uint32_t num_names,
    uint32_t name_type_mask, lldb::LanguageType symbol_language,
    lldb::addr_t offset, SBFileSpecListRef module_list,
    SBFileSpecListRef comp_unit_list) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBBreakpointRef>(
      new SBBreakpoint(unwrapped->BreakpointCreateByNames(
          symbol_name, num_names, name_type_mask, symbol_language, offset,
          *reinterpret_cast<SBFileSpecList *>(module_list),
          *reinterpret_cast<SBFileSpecList *>(comp_unit_list))));
}

SBBreakpointRef SBTargetBreakpointCreateByRegex(SBTargetRef instance,
                                                const char *symbol_name_regex,
                                                const char *module_name) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBBreakpointRef>(new SBBreakpoint(
      unwrapped->BreakpointCreateByRegex(symbol_name_regex, module_name)));
}

SBBreakpointRef SBTargetBreakpointCreateByRegex2(
    SBTargetRef instance, const char *symbol_name_regex,
    SBFileSpecListRef module_list, SBFileSpecListRef comp_unit_list) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBBreakpointRef>(
      new SBBreakpoint(unwrapped->BreakpointCreateByRegex(
          symbol_name_regex, *reinterpret_cast<SBFileSpecList *>(module_list),
          *reinterpret_cast<SBFileSpecList *>(comp_unit_list))));
}

SBBreakpointRef SBTargetBreakpointCreateByRegex3(
    SBTargetRef instance, const char *symbol_name_regex,
    lldb::LanguageType symbol_language, SBFileSpecListRef module_list,
    SBFileSpecListRef comp_unit_list) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBBreakpointRef>(
      new SBBreakpoint(unwrapped->BreakpointCreateByRegex(
          symbol_name_regex, symbol_language,
          *reinterpret_cast<SBFileSpecList *>(module_list),
          *reinterpret_cast<SBFileSpecList *>(comp_unit_list))));
}

SBBreakpointRef SBTargetBreakpointCreateBySourceRegex(SBTargetRef instance,
                                                      const char *source_regex,
                                                      SBFileSpecRef source_file,
                                                      const char *module_name) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBBreakpointRef>(
      new SBBreakpoint(unwrapped->BreakpointCreateBySourceRegex(
          source_regex, *reinterpret_cast<SBFileSpec *>(source_file),
          module_name)));
}

SBBreakpointRef SBTargetBreakpointCreateBySourceRegex2(
    SBTargetRef instance, const char *source_regex,
    SBFileSpecListRef module_list, SBFileSpecListRef source_file) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBBreakpointRef>(
      new SBBreakpoint(unwrapped->BreakpointCreateBySourceRegex(
          source_regex, *reinterpret_cast<SBFileSpecList *>(module_list),
          *reinterpret_cast<SBFileSpecList *>(source_file))));
}

SBBreakpointRef SBTargetBreakpointCreateBySourceRegex3(
    SBTargetRef instance, const char *source_regex,
    SBFileSpecListRef module_list, SBFileSpecListRef source_file,
    SBStringListRef func_names) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBBreakpointRef>(
      new SBBreakpoint(unwrapped->BreakpointCreateBySourceRegex(
          source_regex, *reinterpret_cast<SBFileSpecList *>(module_list),
          *reinterpret_cast<SBFileSpecList *>(source_file),
          *reinterpret_cast<SBStringList *>(func_names))));
}

SBBreakpointRef
SBTargetBreakpointCreateForException(SBTargetRef instance,
                                     lldb::LanguageType language, bool catch_bp,
                                     bool throw_bp) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBBreakpointRef>(new SBBreakpoint(
      unwrapped->BreakpointCreateForException(language, catch_bp, throw_bp)));
}

SBBreakpointRef SBTargetBreakpointCreateByAddress(SBTargetRef instance,
                                                  lldb_addr_t address) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBBreakpointRef>(
      new SBBreakpoint(unwrapped->BreakpointCreateByAddress(address)));
}

SBBreakpointRef SBTargetBreakpointCreateBySBAddress(SBTargetRef instance,
                                                    SBAddressRef address) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBBreakpointRef>(
      new SBBreakpoint(unwrapped->BreakpointCreateBySBAddress(
          *reinterpret_cast<SBAddress *>(address))));
}

SBBreakpointRef SBTargetBreakpointCreateFromScript(
    SBTargetRef instance, const char *class_name,
    SBStructuredDataRef extra_args, SBFileSpecListRef module_list,
    SBFileSpecListRef file_list, bool request_hardware) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBBreakpointRef>(
      new SBBreakpoint(unwrapped->BreakpointCreateFromScript(
          class_name, *reinterpret_cast<SBStructuredData *>(extra_args),
          *reinterpret_cast<SBFileSpecList *>(module_list),
          *reinterpret_cast<SBFileSpecList *>(file_list), request_hardware)));
}

SBErrorRef SBTargetBreakpointsCreateFromFile(SBTargetRef instance,
                                             SBFileSpecRef source_file,
                                             SBBreakpointListRef new_bps) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBErrorRef>(
      new SBError(unwrapped->BreakpointsCreateFromFile(
          *reinterpret_cast<SBFileSpec *>(source_file),
          *reinterpret_cast<SBBreakpointList *>(new_bps))));
}

SBErrorRef SBTargetBreakpointsCreateFromFile2(SBTargetRef instance,
                                              SBFileSpecRef source_file,
                                              SBStringListRef matching_names,
                                              SBBreakpointListRef new_bps) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBErrorRef>(
      new SBError(unwrapped->BreakpointsCreateFromFile(
          *reinterpret_cast<SBFileSpec *>(source_file),
          *reinterpret_cast<SBStringList *>(matching_names),
          *reinterpret_cast<SBBreakpointList *>(new_bps))));
}

SBErrorRef SBTargetBreakspointsWriteToFile(SBTargetRef instance,
                                           SBFileSpecRef dest_file) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBErrorRef>(
      new SBError(unwrapped->BreakpointsWriteToFile(
          *reinterpret_cast<SBFileSpec *>(dest_file))));
}

SBErrorRef SBTargetBreakspointsWriteToFile2(SBTargetRef instance,
                                            SBFileSpecRef dest_file,
                                            SBBreakpointListRef bkpt_list,
                                            bool append) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBErrorRef>(
      new SBError(unwrapped->BreakpointsWriteToFile(
          *reinterpret_cast<SBFileSpec *>(dest_file),
          *reinterpret_cast<SBBreakpointList *>(bkpt_list), append)));
}

uint32_t SBTargetGetNumBreakpoints(SBTargetRef instance) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return unwrapped->GetNumBreakpoints();
}

SBBreakpointRef SBTargetGetBreakpointAtIndex(SBTargetRef instance,
                                             uint32_t idx) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBBreakpointRef>(
      new SBBreakpoint(unwrapped->GetBreakpointAtIndex(idx)));
}

bool SBTargetBreakpointDelete(SBTargetRef instance, lldb_break_id_t break_id) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return unwrapped->BreakpointDelete(break_id);
}

SBBreakpointRef SBTargetFindBreakpointByID(SBTargetRef instance,
                                           lldb_break_id_t break_id) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBBreakpointRef>(
      new SBBreakpoint(unwrapped->FindBreakpointByID(break_id)));
}

bool SBTargetFindBreakpointsByName(SBTargetRef instance, const char *name,
                                   SBBreakpointListRef bkpt_list) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return unwrapped->FindBreakpointsByName(
      name, *reinterpret_cast<SBBreakpointList *>(bkpt_list));
}

void SBTargetGetBreakpointNames(SBTargetRef instance, SBStringListRef names) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  unwrapped->GetBreakpointNames(*reinterpret_cast<SBStringList *>(names));
}

void SBTargetDeleteBreakpointName(SBTargetRef instance, const char *name) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  unwrapped->DeleteBreakpointName(name);
}

bool SBTargetEnableAllBreakpoints(SBTargetRef instance) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return unwrapped->EnableAllBreakpoints();
}

bool SBTargetDisableAllBreakpoints(SBTargetRef instance) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return unwrapped->DisableAllBreakpoints();
}

bool SBTargetDeleteAllBreakpoints(SBTargetRef instance) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return unwrapped->DeleteAllBreakpoints();
}

uint32_t SBTargetGetNumWatchpoints(SBTargetRef instance) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return unwrapped->GetNumWatchpoints();
}

SBWatchpointRef SBTargetGetWatchpointAtIndex(SBTargetRef instance,
                                             uint32_t idx) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBWatchpointRef>(
      new SBWatchpoint(unwrapped->GetWatchpointAtIndex(idx)));
}

bool SBTargetDeleteWatchpoint(SBTargetRef instance, lldb_watch_id_t watch_id) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return unwrapped->DeleteWatchpoint(watch_id);
}

SBWatchpointRef SBTargetFindWatchpointByID(SBTargetRef instance,
                                           lldb_watch_id_t watch_id) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBWatchpointRef>(
      new SBWatchpoint(unwrapped->FindWatchpointByID(watch_id)));
}

SBWatchpointRef SBTargetWatchAddress(SBTargetRef instance, lldb_addr_t addr,
                                     size_t size, bool read, bool write,
                                     SBErrorRef error) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBWatchpointRef>(
      new SBWatchpoint(unwrapped->WatchAddress(
          addr, size, read, write, *reinterpret_cast<SBError *>(error))));
}

bool SBTargetEnableAllWatchpoints(SBTargetRef instance) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return unwrapped->EnableAllWatchpoints();
}

bool SBTargetDisableAllWatchpoints(SBTargetRef instance) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return unwrapped->DisableAllWatchpoints();
}

bool SBTargetDeleteAllWatchpoints(SBTargetRef instance) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return unwrapped->DeleteAllWatchpoints();
}

SBBroadcasterRef SBTargetGetBroadcaster(SBTargetRef instance) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBBroadcasterRef>(
      new SBBroadcaster(unwrapped->GetBroadcaster()));
}

SBTypeRef SBTargetFindFirstType(SBTargetRef instance, const char *type) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBTypeRef>(
      new SBType(unwrapped->FindFirstType(type)));
}

SBTypeListRef SBTargetFindTypes(SBTargetRef instance, const char *type) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBTypeListRef>(
      new SBTypeList(unwrapped->FindTypes(type)));
}

SBTypeRef SBTargetGetBasicType(SBTargetRef instance, lldb::BasicType type) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBTypeRef>(new SBType(unwrapped->GetBasicType(type)));
}

SBValueRef SBTargetCreateValueFromAddress(SBTargetRef instance,
                                          const char *name, SBAddressRef addr,
                                          SBTypeRef type) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBValueRef>(new SBValue(
      unwrapped->CreateValueFromAddress(name,
                                        *reinterpret_cast<SBAddress *>(addr),
                                        *reinterpret_cast<SBType *>(type))));
}

SBValueRef SBTargetCreateValueFromData(SBTargetRef instance, const char *name,
                                       SBDataRef data, SBTypeRef type) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBValueRef>(new SBValue(
      unwrapped->CreateValueFromData(name, *reinterpret_cast<SBData *>(data),
                                     *reinterpret_cast<SBType *>(type))));
}

SBValueRef SBTargetCreateValueFromExpression(SBTargetRef instance,
                                             const char *name,
                                             const char *expr) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBValueRef>(
      new SBValue(unwrapped->CreateValueFromExpression(name, expr)));
}

SBSourceManagerRef SBTargetGetSourceManager(SBTargetRef instance) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBSourceManagerRef>(
      new SBSourceManager(unwrapped->GetSourceManager()));
}

SBInstructionListRef SBTargetReadInstructions(SBTargetRef instance,
                                              SBAddressRef base_addr,
                                              uint32_t count) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBInstructionListRef>(
      new SBInstructionList(unwrapped->ReadInstructions(
          *reinterpret_cast<SBAddress *>(base_addr), count)));
}

SBInstructionListRef SBTargetReadInstructions2(SBTargetRef instance,
                                               SBAddressRef base_addr,
                                               uint32_t count,
                                               const char *flavor_string) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBInstructionListRef>(
      new SBInstructionList(unwrapped->ReadInstructions(
          *reinterpret_cast<SBAddress *>(base_addr), count, flavor_string)));
}

SBInstructionListRef SBTargetGetInstructions(SBTargetRef instance,
                                             SBAddressRef base_addr, void *buf,
                                             size_t size) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBInstructionListRef>(
      new SBInstructionList(unwrapped->GetInstructions(
          *reinterpret_cast<SBAddress *>(base_addr), buf, size)));
}

SBInstructionListRef
SBTargetGetInstructionsWithFlavor(SBTargetRef instance, SBAddressRef base_addr,
                                  const char *flavor_string, void *buf,
                                  size_t size) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBInstructionListRef>(
      new SBInstructionList(unwrapped->GetInstructionsWithFlavor(
          *reinterpret_cast<SBAddress *>(base_addr), flavor_string, buf,
          size)));
}

SBInstructionListRef SBTargetGetInstructions2(SBTargetRef instance,
                                              lldb_addr_t base_addr, void *buf,
                                              size_t size) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBInstructionListRef>(
      new SBInstructionList(unwrapped->GetInstructions(base_addr, buf, size)));
}

SBInstructionListRef
SBTargetGetInstructionsWithFlavor2(SBTargetRef instance, lldb_addr_t base_addr,
                                   const char *flavor_string, void *buf,
                                   size_t size) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBInstructionListRef>(
      new SBInstructionList(unwrapped->GetInstructionsWithFlavor(
          base_addr, flavor_string, buf, size)));
}

SBSymbolContextListRef SBTargetFindSymbols(SBTargetRef instance,
                                           const char *name,
                                           lldb::SymbolType type) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBSymbolContextListRef>(
      new SBSymbolContextList(unwrapped->FindSymbols(name, type)));
}

bool SBTargetGetDescription(SBTargetRef instance, SBStreamRef description,
                            lldb::DescriptionLevel description_level) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description),
                                   description_level);
}

SBValueRef SBTargetEvaluateExpression(SBTargetRef instance, const char *expr,
                                      SBExpressionOptionsRef options) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBValueRef>(new SBValue(unwrapped->EvaluateExpression(
      expr, *reinterpret_cast<SBExpressionOptions *>(options))));
}

lldb::addr_t SBTargetGetStackRedZoneSize(SBTargetRef instance) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return unwrapped->GetStackRedZoneSize();
}

bool SBTargetIsLoaded(SBTargetRef instance, SBModuleRef module) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return unwrapped->IsLoaded(*reinterpret_cast<SBModule *>(module));
}

SBLaunchInfoRef SBTargetGetLaunchInfo(SBTargetRef instance) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  return reinterpret_cast<SBLaunchInfoRef>(
      new SBLaunchInfo(unwrapped->GetLaunchInfo()));
}

void SBTargetSetLaunchInfo(SBTargetRef instance, SBLaunchInfoRef launch_info) {
  SBTarget *unwrapped = reinterpret_cast<SBTarget *>(instance);
  unwrapped->SetLaunchInfo(*reinterpret_cast<SBLaunchInfo *>(launch_info));
}

#ifdef __cplusplus
}
#endif
