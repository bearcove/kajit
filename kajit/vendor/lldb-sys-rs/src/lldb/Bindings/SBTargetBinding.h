//===-- SBTargetBinding.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBTargetBinding_h_
#define LLDB_SBTargetBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBTargetRef CreateSBTarget(void);

LLDB_API SBTargetRef CloneSBTarget(SBTargetRef instance);

LLDB_API void DisposeSBTarget(SBTargetRef instance);

LLDB_API bool SBTargetIsValid(SBTargetRef instance);

LLDB_API bool SBTargetEventIsTargetEvent(SBEventRef event);

LLDB_API SBTargetRef SBTargetGetTargetFromEvent(SBEventRef event);

LLDB_API uint32_t SBTargetGetNumModulesFromEvent(SBEventRef event);

LLDB_API SBModuleRef SBTargetGetModuleAtIndexFromEvent(const uint32_t idx,
                                                       SBEventRef event);

LLDB_API const char *SBTargetGetBroadcasterClassName(void);

LLDB_API SBProcessRef SBTargetGetProcess(SBTargetRef instance);

LLDB_API void SBTargetSetCollectingStats(SBTargetRef instance, bool v);

LLDB_API bool SBTargetGetCollectingStats(SBTargetRef instance);

LLDB_API SBStructuredDataRef SBTargetGetStatistics(SBTargetRef instance);

LLDB_API SBPlatformRef SBTargetGetPlatform(SBTargetRef instance);

LLDB_API SBEnvironmentRef SBTargetGetEnvironment(SBTargetRef instance);

LLDB_API SBErrorRef SBTargetInstall(SBTargetRef instance);

LLDB_API SBProcessRef SBTargetLaunch(
    SBTargetRef instance, SBListenerRef listener, const char **argv,
    const char **envp, const char *stdin_path, const char *stdout_path,
    const char *stderr_path, const char *working_directory,
    uint32_t launch_flags, bool stop_at_entry, SBErrorRef error);

LLDB_API SBProcessRef SBTargetLaunchSimple(SBTargetRef instance,
                                           const char **argv, const char **envp,
                                           const char *working_directory);

LLDB_API SBProcessRef SBTargetLaunch2(SBTargetRef instance,
                                      SBLaunchInfoRef launch_info,
                                      SBErrorRef error);

LLDB_API SBProcessRef SBTargetLoadCore(SBTargetRef instance,
                                       const char *core_file, SBErrorRef error);

LLDB_API SBProcessRef SBTargetAttach(SBTargetRef instance,
                                     SBAttachInfoRef attach_info,
                                     SBErrorRef error);

LLDB_API SBProcessRef SBTargetAttachToProcessWithID(SBTargetRef instance,
                                                    SBListenerRef listener,
                                                    lldb_pid_t pid,
                                                    SBErrorRef error);

LLDB_API SBProcessRef SBTargetAttachToProcessWithName(SBTargetRef instance,
                                                      SBListenerRef listener,
                                                      const char *name,
                                                      bool wait_for,
                                                      SBErrorRef error);

LLDB_API SBProcessRef SBTargetConnectRemote(SBTargetRef instance,
                                            SBListenerRef listener,
                                            const char *url,
                                            const char *plugin_name,
                                            SBErrorRef error);

LLDB_API SBFileSpecRef SBTargetGetExecutable(SBTargetRef instance);

LLDB_API void SBTargetAppendImageSearchPath(SBTargetRef instance,
                                            const char *from, const char *to,
                                            SBErrorRef error);

LLDB_API bool SBTargetAddModule(SBTargetRef instance, SBModuleRef module);

LLDB_API SBModuleRef SBTargetAddModuleSpec(SBTargetRef instance,
                                           SBModuleSpecRef module_spec);

LLDB_API uint32_t SBTargetGetNumModules(SBTargetRef instance);

LLDB_API SBModuleRef SBTargetGetModuleAtIndex(SBTargetRef instance,
                                              uint32_t idx);

LLDB_API bool SBTargetRemoveModule(SBTargetRef instance, SBModuleRef module);

LLDB_API SBDebuggerRef SBTargetGetDebugger(SBTargetRef instance);

LLDB_API SBModuleRef SBTargetFindModule(SBTargetRef instance,
                                        SBFileSpecRef file_spec);

LLDB_API SBSymbolContextListRef
SBTargetFindCompileUnits(SBTargetRef instance, SBFileSpecRef file_spec);

LLDB_API ENUM(ByteOrder) SBTargetGetByteOrder(SBTargetRef instance);

LLDB_API uint32_t SBTargetGetAddressByteSize(SBTargetRef instance);

LLDB_API const char *SBTargetGetTriple(SBTargetRef instance);

LLDB_API const char *SBTargetGetABIName(SBTargetRef instance);

LLDB_API uint32_t SBTargetGetDataByteSize(SBTargetRef instance);

LLDB_API uint32_t SBTargetGetCodeByteSize(SBTargetRef instance);

LLDB_API uint32_t
SBTargetGetMaximumNumberOfChildrenToDisplay(SBTargetRef instance);

LLDB_API SBErrorRef SBTargetSetSectionLoadAddress(
    SBTargetRef instance, SBSectionRef section, lldb_addr_t section_base_addr);

LLDB_API SBErrorRef SBTargetClearSectionLoadAddress(SBTargetRef instance,
                                                    SBSectionRef section);

LLDB_API SBErrorRef SBTargetSetModuleLoadAddress(SBTargetRef instance,
                                                 SBModuleRef module,
                                                 int64_t sections_offset);

LLDB_API SBErrorRef SBTargetClearModuleLoadAddress(SBTargetRef instance,
                                                   SBModuleRef module);

LLDB_API SBSymbolContextListRef SBTargetFindFunctions(SBTargetRef instance,
                                                      const char *name,
                                                      uint32_t name_type_mask);

LLDB_API SBValueListRef SBTargetFindGlobalVariables(SBTargetRef instance,
                                                    const char *name,
                                                    uint32_t max_matches);

LLDB_API SBValueRef SBTargetFindFirstGlobalVariable(SBTargetRef instance,
                                                    const char *name);

LLDB_API SBValueListRef SBTargetFindGlobalVariables2(SBTargetRef instance,
                                                     const char *name,
                                                     uint32_t max_matches,
                                                     ENUM(MatchType) matchtype);

LLDB_API SBSymbolContextListRef
SBTargetFindGlobalFunctions(SBTargetRef instance, const char *name,
                            uint32_t max_matches, ENUM(MatchType) matchtype);

LLDB_API void SBTargetClear(SBTargetRef instance);

LLDB_API SBAddressRef SBTargetResolveFileAddress(SBTargetRef instance,
                                                 lldb_addr_t file_addr);

LLDB_API SBAddressRef SBTargetResolveLoadAddress(SBTargetRef instance,
                                                 lldb_addr_t vm_addr);

LLDB_API SBAddressRef SBTargetResolvePastLoadAddress(SBTargetRef instance,
                                                     uint32_t stop_id,
                                                     lldb_addr_t vm_addr);

LLDB_API SBSymbolContextRef SBTargetResolveSymbolContextForAddress(
    SBTargetRef instance, SBAddressRef addr, uint32_t resolve_scope);

LLDB_API size_t SBTargetReadMemory(SBTargetRef instance, SBAddressRef addr,
                                   void *buf, size_t size, SBErrorRef error);

LLDB_API SBBreakpointRef SBTargetBreakpointCreateByLocation(
    SBTargetRef instance, const char *file, uint32_t line);

LLDB_API SBBreakpointRef SBTargetBreakpointCreateByLocation2(
    SBTargetRef instance, SBFileSpecRef file_spec, uint32_t line);

LLDB_API SBBreakpointRef SBTargetBreakpointCreateByLocation3(
    SBTargetRef instance, SBFileSpecRef file_spec, uint32_t line,
    lldb_addr_t offset);

LLDB_API SBBreakpointRef SBTargetBreakpointCreateByLocation4(
    SBTargetRef instance, SBFileSpecRef file_spec, uint32_t line,
    lldb_addr_t offset, SBFileSpecListRef module_list);

LLDB_API SBBreakpointRef SBTargetBreakpointCreateByLocation5(
    SBTargetRef instance, SBFileSpecRef file_spec, uint32_t line,
    uint32_t column, lldb_addr_t offset, SBFileSpecListRef module_list);

LLDB_API SBBreakpointRef SBTargetBreakpointCreateByLocation6(
    SBTargetRef instance, SBFileSpecRef file_spec, uint32_t line,
    uint32_t column, lldb_addr_t offset, SBFileSpecListRef module_list,
    bool move_to_nearest_code);

LLDB_API SBBreakpointRef SBTargetBreakpointCreateByName(
    SBTargetRef instance, const char *symbol_name, const char *module_name);

LLDB_API SBBreakpointRef SBTargetBreakpointCreateByName2(
    SBTargetRef instance, const char *symbol_name,
    SBFileSpecListRef module_list, SBFileSpecListRef comp_unit_list);

LLDB_API SBBreakpointRef SBTargetBreakpointCreateByName3(
    SBTargetRef instance, const char *symbol_name, uint32_t name_type_mask,
    SBFileSpecListRef module_list, SBFileSpecListRef comp_unit_list);

LLDB_API SBBreakpointRef SBTargetBreakpointCreateByNames(
    SBTargetRef instance, const char **symbol_name, uint32_t num_names,
    uint32_t name_type_mask, SBFileSpecListRef module_list,
    SBFileSpecListRef comp_unit_list);

LLDB_API SBBreakpointRef SBTargetBreakpointCreateByNames2(
    SBTargetRef instance, const char **symbol_name, uint32_t num_names,
    uint32_t name_type_mask, ENUM(LanguageType) symbol_language,
    SBFileSpecListRef module_list, SBFileSpecListRef comp_unit_list);

LLDB_API SBBreakpointRef SBTargetBreakpointCreateByNames3(
    SBTargetRef instance, const char **symbol_name, uint32_t num_names,
    uint32_t name_type_mask, ENUM(LanguageType) symbol_language,
    lldb_addr_t offset, SBFileSpecListRef module_list,
    SBFileSpecListRef comp_unit_list);

LLDB_API SBBreakpointRef SBTargetBreakpointCreateByRegex(
    SBTargetRef instance, const char *symbol_name_regex,
    const char *module_name);

LLDB_API SBBreakpointRef SBTargetBreakpointCreateByRegex2(
    SBTargetRef instance, const char *symbol_name_regex,
    SBFileSpecListRef module_list, SBFileSpecListRef comp_unit_list);

LLDB_API SBBreakpointRef SBTargetBreakpointCreateByRegex3(
    SBTargetRef instance, const char *symbol_name_regex,
    ENUM(LanguageType) symbol_language, SBFileSpecListRef module_list,
    SBFileSpecListRef comp_unit_list);

LLDB_API SBBreakpointRef SBTargetBreakpointCreateBySourceRegex(
    SBTargetRef instance, const char *source_regex, SBFileSpecRef source_file,
    const char *module_name);

LLDB_API SBBreakpointRef SBTargetBreakpointCreateBySourceRegex2(
    SBTargetRef instance, const char *source_regex,
    SBFileSpecListRef module_list, SBFileSpecListRef source_file);

LLDB_API SBBreakpointRef SBTargetBreakpointCreateBySourceRegex3(
    SBTargetRef instance, const char *source_regex,
    SBFileSpecListRef module_list, SBFileSpecListRef source_file,
    SBStringListRef func_names);

LLDB_API SBBreakpointRef SBTargetBreakpointCreateForException(
    SBTargetRef instance, ENUM(LanguageType) language, bool catch_bp,
    bool throw_bp);

LLDB_API SBBreakpointRef SBTargetBreakpointCreateByAddress(SBTargetRef instance,
                                                           lldb_addr_t address);

LLDB_API SBBreakpointRef
SBTargetBreakpointCreateBySBAddress(SBTargetRef instance, SBAddressRef address);

LLDB_API SBBreakpointRef SBTargetBreakpointCreateFromScript(
    SBTargetRef instance, const char *class_name,
    SBStructuredDataRef extra_args, SBFileSpecListRef module_list,
    SBFileSpecListRef file_list, bool request_hardware);
LLDB_API SBErrorRef SBTargetBreakpointsCreateFromFile(
    SBTargetRef instance, SBFileSpecRef source_file,
    SBBreakpointListRef new_bps);

LLDB_API SBErrorRef SBTargetBreakpointsCreateFromFile2(
    SBTargetRef instance, SBFileSpecRef source_file,
    SBStringListRef matching_names, SBBreakpointListRef new_bps);

LLDB_API SBErrorRef SBTargetBreakspointsWriteToFile(SBTargetRef instance,
                                                    SBFileSpecRef dest_file);

LLDB_API SBErrorRef
SBTargetBreakspointsWriteToFile2(SBTargetRef instance, SBFileSpecRef dest_file,
                                 SBBreakpointListRef bkpt_list, bool append);

LLDB_API uint32_t SBTargetGetNumBreakpoints(SBTargetRef instance);

LLDB_API SBBreakpointRef SBTargetGetBreakpointAtIndex(SBTargetRef instance,
                                                      uint32_t idx);

LLDB_API bool SBTargetBreakpointDelete(SBTargetRef instance,
                                       lldb_break_id_t break_id);

LLDB_API SBBreakpointRef SBTargetFindBreakpointByID(SBTargetRef instance,
                                                    lldb_break_id_t break_id);

LLDB_API bool SBTargetFindBreakpointsByName(SBTargetRef instance,
                                            const char *name,
                                            SBBreakpointListRef bkpt_list);

LLDB_API void SBTargetGetBreakpointNames(SBTargetRef instance,
                                         SBStringListRef names);

LLDB_API void SBTargetDeleteBreakpointName(SBTargetRef instance,
                                           const char *name);

LLDB_API bool SBTargetEnableAllBreakpoints(SBTargetRef instance);

LLDB_API bool SBTargetDisableAllBreakpoints(SBTargetRef instance);

LLDB_API bool SBTargetDeleteAllBreakpoints(SBTargetRef instance);

LLDB_API uint32_t SBTargetGetNumWatchpoints(SBTargetRef instance);

LLDB_API SBWatchpointRef SBTargetGetWatchpointAtIndex(SBTargetRef instance,
                                                      uint32_t idx);

LLDB_API bool SBTargetDeleteWatchpoint(SBTargetRef instance,
                                       lldb_watch_id_t watch_id);

LLDB_API SBWatchpointRef SBTargetFindWatchpointByID(SBTargetRef instance,
                                                    lldb_watch_id_t watch_id);

LLDB_API SBWatchpointRef SBTargetWatchAddress(SBTargetRef instance,
                                              lldb_addr_t addr, size_t size,
                                              bool read, bool write,
                                              SBErrorRef error);

LLDB_API bool SBTargetEnableAllWatchpoints(SBTargetRef instance);

LLDB_API bool SBTargetDisableAllWatchpoints(SBTargetRef instance);

LLDB_API bool SBTargetDeleteAllWatchpoints(SBTargetRef instance);

LLDB_API SBBroadcasterRef SBTargetGetBroadcaster(SBTargetRef instance);

LLDB_API SBTypeRef SBTargetFindFirstType(SBTargetRef instance,
                                         const char *type);

LLDB_API SBTypeListRef SBTargetFindTypes(SBTargetRef instance,
                                         const char *type);

LLDB_API SBTypeRef SBTargetGetBasicType(SBTargetRef instance,
                                        ENUM(BasicType) type);

LLDB_API SBValueRef SBTargetCreateValueFromAddress(SBTargetRef instance,
                                                   const char *name,
                                                   SBAddressRef addr,
                                                   SBTypeRef type);

LLDB_API SBValueRef SBTargetCreateValueFromData(SBTargetRef instance,
                                                const char *name,
                                                SBDataRef data, SBTypeRef type);

LLDB_API SBValueRef SBTargetCreateValueFromExpression(SBTargetRef instance,
                                                      const char *name,
                                                      const char *expr);

LLDB_API SBSourceManagerRef SBTargetGetSourceManager(SBTargetRef instance);

LLDB_API SBInstructionListRef SBTargetReadInstructions(SBTargetRef instance,
                                                       SBAddressRef base_addr,
                                                       uint32_t count);

LLDB_API SBInstructionListRef
SBTargetReadInstructions2(SBTargetRef instance, SBAddressRef base_addr,
                          uint32_t count, const char *flavor_string);

LLDB_API SBInstructionListRef SBTargetGetInstructions(SBTargetRef instance,
                                                      SBAddressRef base_addr,
                                                      void *buf, size_t size);

LLDB_API SBInstructionListRef SBTargetGetInstructionsWithFlavor(
    SBTargetRef instance, SBAddressRef base_addr, const char *flavor_string,
    void *buf, size_t size);

LLDB_API SBInstructionListRef SBTargetGetInstructions2(SBTargetRef instance,
                                                       lldb_addr_t base_addr,
                                                       void *buf, size_t size);

LLDB_API SBInstructionListRef SBTargetGetInstructionsWithFlavor2(
    SBTargetRef instance, lldb_addr_t base_addr, const char *flavor_string,
    void *buf, size_t size);

LLDB_API SBSymbolContextListRef SBTargetFindSymbols(SBTargetRef instance,
                                                    const char *name,
                                                    ENUM(SymbolType) type);

LLDB_API bool SBTargetGetDescription(SBTargetRef instance,
                                     SBStreamRef description,
                                     ENUM(DescriptionLevel) description_level);

LLDB_API SBValueRef SBTargetEvaluateExpression(SBTargetRef instance,
                                               const char *expr,
                                               SBExpressionOptionsRef options);

LLDB_API lldb_addr_t SBTargetGetStackRedZoneSize(SBTargetRef instance);

LLDB_API bool SBTargetIsLoaded(SBTargetRef instance, SBModuleRef module);

LLDB_API SBLaunchInfoRef SBTargetGetLaunchInfo(SBTargetRef instance);

LLDB_API void SBTargetSetLaunchInfo(SBTargetRef instance,
                                    SBLaunchInfoRef launch_info);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBTargetBinding_h_
