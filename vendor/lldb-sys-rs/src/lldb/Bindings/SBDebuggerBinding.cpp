//===-- SBDebuggerBinding.cpp -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/LLDB.h"
#include "lldb/API/SBTypeCategory.h"
#include "lldb/API/SBTypeFilter.h"
#include "lldb/API/SBTypeFormat.h"
#include "lldb/API/SBTypeNameSpecifier.h"
#include "lldb/API/SBTypeSummary.h"
#include "lldb/API/SBTypeSynthetic.h"
#include "lldb/Bindings/LLDBBinding.h"

using namespace lldb;

#ifdef __cplusplus
extern "C" {
#endif

SBInputReaderRef CreateSBInputReader() {
  return reinterpret_cast<SBInputReaderRef>(new SBInputReader());
}

SBInputReaderRef CloneSBInputReader(SBInputReaderRef instance) {
  return reinterpret_cast<SBInputReaderRef>(
      new SBInputReader(*reinterpret_cast<SBInputReader *>(instance)));
}

void DisposeSBInputReader(SBInputReaderRef instance) {
  delete reinterpret_cast<SBInputReader *>(instance);
}

SBErrorRef SBInputReaderInitialize(SBInputReaderRef instance,
                                   SBDebuggerRef var0, Callback callback,
                                   void *callback_baton,
                                   lldb::InputReaderGranularity var1,
                                   const char *var2, const char *var3,
                                   bool var4) {
  SBInputReader *unwrapped = reinterpret_cast<SBInputReader *>(instance);
  return reinterpret_cast<SBErrorRef>(new SBError(
      unwrapped->Initialize(*reinterpret_cast<SBDebugger *>(var0), callback,
                            callback_baton, var1, var2, var3, var4)));
}

void SBInputReaderSetIsDone(SBInputReaderRef instance, bool var0) {
  SBInputReader *unwrapped = reinterpret_cast<SBInputReader *>(instance);
  unwrapped->SetIsDone(var0);
}

bool SBInputReaderIsActive(SBInputReaderRef instance) {
  SBInputReader *unwrapped = reinterpret_cast<SBInputReader *>(instance);
  return unwrapped->IsActive();
}

const char *SBDebuggerGetBroadcasterClass() {
  return lldb::SBProcess::GetBroadcasterClass();
}

SBBroadcasterRef SBDebuggerGetBroadcaster(SBDebuggerRef instance) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return reinterpret_cast<SBBroadcasterRef>(
      new SBBroadcaster(unwrapped->GetBroadcaster()));
}

const char *SBDebuggerGetProgressFromEvent(SBEventRef event,
                                           uint64_t *progress_id,
                                           uint64_t *completed, uint64_t *total,
                                           bool *is_debugger_specific) {
  return SBDebugger::GetProgressFromEvent(*reinterpret_cast<SBEvent *>(event),
                                          *progress_id, *completed, *total,
                                          *is_debugger_specific);
}

void SBDebuggerInitialize() { lldb::SBDebugger::Initialize(); }

void SBDebuggerTerminate() { lldb::SBDebugger::Terminate(); }

SBDebuggerRef SBDebuggerCreate() {
  return reinterpret_cast<SBDebuggerRef>(
      new SBDebugger(lldb::SBDebugger::Create()));
}

SBDebuggerRef SBDebuggerCreate2(bool source_init_files) {
  return reinterpret_cast<SBDebuggerRef>(
      new SBDebugger(lldb::SBDebugger::Create(source_init_files)));
}

SBDebuggerRef SBDebuggerCreate3(bool source_init_files,
                                LogOutputCallback log_callback, void *baton) {
  return reinterpret_cast<SBDebuggerRef>(new SBDebugger(
      lldb::SBDebugger::Create(source_init_files, log_callback, baton)));
}

void SBDebuggerDestroy(SBDebuggerRef debugger) {
  lldb::SBDebugger::Destroy(*reinterpret_cast<SBDebugger *>(debugger));
}

void SBDebuggerMemoryPressureDetected() {
  lldb::SBDebugger::MemoryPressureDetected();
}

SBDebuggerRef CreateSBDebugger() {
  return reinterpret_cast<SBDebuggerRef>(new SBDebugger());
}

SBDebuggerRef CloneSBDebugger(SBDebuggerRef instance) {
  return reinterpret_cast<SBDebuggerRef>(
      new SBDebugger(*reinterpret_cast<SBDebugger *>(instance)));
}

void DisposeSBDebugger(SBDebuggerRef instance) {
  delete reinterpret_cast<SBDebugger *>(instance);
}

bool SBDebuggerIsValid(SBDebuggerRef instance) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return unwrapped->IsValid();
}

void SBDebuggerClear(SBDebuggerRef instance) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  unwrapped->Clear();
}

void SBDebuggerSetAsync(SBDebuggerRef instance, bool b) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  unwrapped->SetAsync(b);
}

bool SBDebuggerGetAsync(SBDebuggerRef instance) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return unwrapped->GetAsync();
}

void SBDebuggerSkipLLDBInitFiles(SBDebuggerRef instance, bool b) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  unwrapped->SkipLLDBInitFiles(b);
}

void SBDebuggerSkipAppInitFiles(SBDebuggerRef instance, bool b) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  unwrapped->SkipAppInitFiles(b);
}

SBErrorRef SBDebuggerSetInputString(SBDebuggerRef instance, const char *data) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return reinterpret_cast<SBErrorRef>(
      new SBError(unwrapped->SetInputString(data)));
}

SBErrorRef SBDebuggerSetInputFile(SBDebuggerRef instance, SBFileRef file) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return reinterpret_cast<SBErrorRef>(
      new SBError(unwrapped->SetInputFile(*reinterpret_cast<SBFile *>(file))));
}

SBErrorRef SBDebuggerSetOutputFile(SBDebuggerRef instance, SBFileRef file) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return reinterpret_cast<SBErrorRef>(
      new SBError(unwrapped->SetOutputFile(*reinterpret_cast<SBFile *>(file))));
}

SBErrorRef SBDebuggerSetErrorFile(SBDebuggerRef instance, SBFileRef file) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return reinterpret_cast<SBErrorRef>(
      new SBError(unwrapped->SetErrorFile(*reinterpret_cast<SBFile *>(file))));
}

SBFileRef SBDebuggerGetInputFile(SBDebuggerRef instance) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return reinterpret_cast<SBFileRef>(new SBFile(unwrapped->GetInputFile()));
}

SBFileRef SBDebuggerGetOutputFile(SBDebuggerRef instance) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return reinterpret_cast<SBFileRef>(new SBFile(unwrapped->GetOutputFile()));
}

SBFileRef SBDebuggerGetErrorFile(SBDebuggerRef instance) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return reinterpret_cast<SBFileRef>(new SBFile(unwrapped->GetErrorFile()));
}

void SBDebuggerSaveInputTerminalState(SBDebuggerRef instance) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  unwrapped->SaveInputTerminalState();
}

void SBDebuggerRestoreInputTerminalState(SBDebuggerRef instance) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  unwrapped->RestoreInputTerminalState();
}

SBCommandInterpreterRef
SBDebuggerGetCommandInterpreter(SBDebuggerRef instance) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return reinterpret_cast<SBCommandInterpreterRef>(
      new SBCommandInterpreter(unwrapped->GetCommandInterpreter()));
}

void SBDebuggerHandleCommand(SBDebuggerRef instance, const char *command) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  unwrapped->HandleCommand(command);
}

SBListenerRef SBDebuggerGetListener(SBDebuggerRef instance) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return reinterpret_cast<SBListenerRef>(
      new SBListener(unwrapped->GetListener()));
}

void SBDebuggerHandleProcessEvent(SBDebuggerRef instance, SBProcessRef process,
                                  SBEventRef event, SBFileRef out,
                                  SBFileRef err) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  unwrapped->HandleProcessEvent(*reinterpret_cast<SBProcess *>(process),
                                *reinterpret_cast<SBEvent *>(event),
                                *reinterpret_cast<SBFile *>(out),
                                *reinterpret_cast<SBFile *>(err));
}

SBTargetRef SBDebuggerCreateTarget(SBDebuggerRef instance, const char *filename,
                                   const char *target_triple,
                                   const char *platform_name,
                                   bool add_dependent_modules,
                                   SBErrorRef error) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return reinterpret_cast<SBTargetRef>(new SBTarget(unwrapped->CreateTarget(
      filename, target_triple, platform_name, add_dependent_modules,
      *reinterpret_cast<SBError *>(error))));
}

SBTargetRef SBDebuggerCreateTargetWithFileAndTargetTriple(
    SBDebuggerRef instance, const char *filename, const char *target_triple) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return reinterpret_cast<SBTargetRef>(new SBTarget(
      unwrapped->CreateTargetWithFileAndTargetTriple(filename, target_triple)));
}

SBTargetRef SBDebuggerCreateTargetWithFileAndArch(SBDebuggerRef instance,
                                                  const char *filename,
                                                  const char *archname) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return reinterpret_cast<SBTargetRef>(
      new SBTarget(unwrapped->CreateTargetWithFileAndArch(filename, archname)));
}

SBTargetRef SBDebuggerCreateTarget2(SBDebuggerRef instance,
                                    const char *filename) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return reinterpret_cast<SBTargetRef>(
      new SBTarget(unwrapped->CreateTarget(filename)));
}

SBTargetRef SBDebuggerGetDummyTarget(SBDebuggerRef instance) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return reinterpret_cast<SBTargetRef>(
      new SBTarget(unwrapped->GetDummyTarget()));
}

bool SBDebuggerDeleteTarget(SBDebuggerRef instance, SBTargetRef target) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return unwrapped->DeleteTarget(*reinterpret_cast<SBTarget *>(target));
}

SBTargetRef SBDebuggerGetTargetAtIndex(SBDebuggerRef instance, uint32_t idx) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return reinterpret_cast<SBTargetRef>(
      new SBTarget(unwrapped->GetTargetAtIndex(idx)));
}

uint32_t SBDebuggerGetIndexOfTarget(SBDebuggerRef instance,
                                    SBTargetRef target) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return unwrapped->GetIndexOfTarget(*reinterpret_cast<SBTarget *>(target));
}

SBTargetRef SBDebuggerFindTargetWithProcessID(SBDebuggerRef instance,
                                              lldb_pid_t pid) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return reinterpret_cast<SBTargetRef>(
      new SBTarget(unwrapped->FindTargetWithProcessID(pid)));
}

SBTargetRef SBDebuggerFindTargetWithFileAndArch(SBDebuggerRef instance,
                                                const char *filename,
                                                const char *arch) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return reinterpret_cast<SBTargetRef>(
      new SBTarget(unwrapped->FindTargetWithFileAndArch(filename, arch)));
}

uint32_t SBDebuggerGetNumTargets(SBDebuggerRef instance) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return unwrapped->GetNumTargets();
}

SBTargetRef SBDebuggerGetSelectedTarget(SBDebuggerRef instance) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return reinterpret_cast<SBTargetRef>(
      new SBTarget(unwrapped->GetSelectedTarget()));
}

void SBDebuggerSetSelectedTarget(SBDebuggerRef instance, SBTargetRef target) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  unwrapped->SetSelectedTarget(*reinterpret_cast<SBTarget *>(target));
}

SBPlatformRef SBDebuggerGetSelectedPlatform(SBDebuggerRef instance) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return reinterpret_cast<SBPlatformRef>(
      new SBPlatform(unwrapped->GetSelectedPlatform()));
}

void SBDebuggerSetSelectedPlatform(SBDebuggerRef instance,
                                   SBPlatformRef platform) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  unwrapped->SetSelectedPlatform(*reinterpret_cast<SBPlatform *>(platform));
}

uint32_t SBDebuggerGetNumPlatforms(SBDebuggerRef instance) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return unwrapped->GetNumPlatforms();
}

SBPlatformRef SBDebuggerGetPlatformAtIndex(SBDebuggerRef instance,
                                           uint32_t idx) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return reinterpret_cast<SBPlatformRef>(
      new SBPlatform(unwrapped->GetPlatformAtIndex(idx)));
}

uint32_t SBDebuggerGetNumAvailablePlatforms(SBDebuggerRef instance) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return unwrapped->GetNumAvailablePlatforms();
}

SBStructuredDataRef
SBDebuggerGetAvailablePlatformInfoAtIndex(SBDebuggerRef instance,
                                          uint32_t idx) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return reinterpret_cast<SBStructuredDataRef>(
      new SBStructuredData(unwrapped->GetAvailablePlatformInfoAtIndex(idx)));
}

SBSourceManagerRef SBDebuggerGetSourceManager(SBDebuggerRef instance) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return reinterpret_cast<SBSourceManagerRef>(
      new SBSourceManager(unwrapped->GetSourceManager()));
}

SBErrorRef SBDebuggerSetCurrentPlatform(SBDebuggerRef instance,
                                        const char *platform_name) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return reinterpret_cast<SBErrorRef>(
      new SBError(unwrapped->SetCurrentPlatform(platform_name)));
}

bool SBDebuggerSetCurrentPlatformSDKRoot(SBDebuggerRef instance,
                                         const char *sysroot) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return unwrapped->SetCurrentPlatformSDKRoot(sysroot);
}

bool SBDebuggerSetUseExternalEditor(SBDebuggerRef instance, bool input) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return unwrapped->SetUseExternalEditor(input);
}

bool SBDebuggerGetUseExternalEditor(SBDebuggerRef instance) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return unwrapped->GetUseExternalEditor();
}

bool SBDebuggerSetUseColor(SBDebuggerRef instance, bool use_color) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return unwrapped->SetUseColor(use_color);
}

bool SBDebuggerGetUseColor(SBDebuggerRef instance) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return unwrapped->GetUseColor();
}

bool SBDebuggerSetUseSourceCache(SBDebuggerRef instance,
                                 bool use_source_cache) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return unwrapped->SetUseSourceCache(use_source_cache);
}

bool SBDebuggerGetUseSourceCache(SBDebuggerRef instance) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return unwrapped->GetUseSourceCache();
}

bool SBDebuggerGetDefaultArchitecture(char *arch_name, size_t arch_name_len) {
  return lldb::SBDebugger::GetDefaultArchitecture(arch_name, arch_name_len);
}

bool SBDebuggerSetDefaultArchitecture(const char *arch_name) {
  return lldb::SBDebugger::SetDefaultArchitecture(arch_name);
}

enum lldb::ScriptLanguage
SBDebuggerGetScriptingLanguage(SBDebuggerRef instance,
                               const char *script_language_name) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return unwrapped->GetScriptingLanguage(script_language_name);
}

SBStructuredDataRef
SBDebuggerGetScriptInterpreterInfo(SBDebuggerRef instance,
                                   lldb::ScriptLanguage language) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return reinterpret_cast<SBStructuredDataRef>(
      new SBStructuredData(unwrapped->GetScriptInterpreterInfo(language)));
}

const char *SBDebuggerGetVersionString() {
  return lldb::SBDebugger::GetVersionString();
}

const char *SBDebuggerStateAsCString(lldb::StateType state) {
  return lldb::SBDebugger::StateAsCString(state);
}

bool SBDebuggerStateIsRunningState(lldb::StateType state) {
  return lldb::SBDebugger::StateIsRunningState(state);
}

bool SBDebuggerStateIsStoppedState(lldb::StateType state) {
  return lldb::SBDebugger::StateIsStoppedState(state);
}

bool SBDebuggerEnableLog(SBDebuggerRef instance, const char *channel,
                         const char **categories) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return unwrapped->EnableLog(channel, categories);
}

void SBDebuggerSetLoggingCallback(SBDebuggerRef instance,
                                  lldb::LogOutputCallback log_callback,
                                  void *baton) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  unwrapped->SetLoggingCallback(log_callback, baton);
}

void SBDebuggerDispatchInput(SBDebuggerRef instance, void *baton,
                             const void *data, size_t data_len) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  unwrapped->DispatchInput(baton, data, data_len);
}

void SBDebuggerDispatchInput2(SBDebuggerRef instance, const void *data,
                              size_t data_len) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  unwrapped->DispatchInput(data, data_len);
}

void SBDebuggerDispatchInputInterrupt(SBDebuggerRef instance) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  unwrapped->DispatchInputInterrupt();
}

void SBDebuggerDispatchInputEndOfFile(SBDebuggerRef instance) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  unwrapped->DispatchInputEndOfFile();
}

void SBDebuggerPushInputReader(SBDebuggerRef instance,
                               SBInputReaderRef reader) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  unwrapped->PushInputReader(*reinterpret_cast<SBInputReader *>(reader));
}

const char *SBDebuggerGetInstanceName(SBDebuggerRef instance) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return unwrapped->GetInstanceName();
}

SBDebuggerRef SBDebuggerFindDebuggerWithID(int id) {
  return reinterpret_cast<SBDebuggerRef>(
      new SBDebugger(lldb::SBDebugger::FindDebuggerWithID(id)));
}

SBErrorRef SBDebuggerSetInternalVariable(const char *var_name,
                                         const char *value,
                                         const char *debugger_instance_name) {
  return reinterpret_cast<SBErrorRef>(
      new SBError(lldb::SBDebugger::SetInternalVariable(
          var_name, value, debugger_instance_name)));
}

SBStringListRef
SBDebuggerGetInternalVariableValue(const char *var_name,
                                   const char *debugger_instance_name) {
  return reinterpret_cast<SBStringListRef>(
      new SBStringList(lldb::SBDebugger::GetInternalVariableValue(
          var_name, debugger_instance_name)));
}

bool SBDebuggerGetDescription(SBDebuggerRef instance, SBStreamRef description) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description));
}

uint32_t SBDebuggerGetTerminalWidth(SBDebuggerRef instance) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return unwrapped->GetTerminalWidth();
}

void SBDebuggerSetTerminalWidth(SBDebuggerRef instance, uint32_t term_width) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  unwrapped->SetTerminalWidth(term_width);
}

lldb_user_id_t SBDebuggerGetID(SBDebuggerRef instance) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return unwrapped->GetID();
}

const char *SBDebuggerGetPrompt(SBDebuggerRef instance) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return unwrapped->GetPrompt();
}

void SBDebuggerSetPrompt(SBDebuggerRef instance, const char *prompt) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  unwrapped->SetPrompt(prompt);
}

enum lldb::ScriptLanguage SBDebuggerGetScriptLanguage(SBDebuggerRef instance) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return unwrapped->GetScriptLanguage();
}

void SBDebuggerSetScriptLanguage(SBDebuggerRef instance,
                                 lldb::ScriptLanguage script_lang) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  unwrapped->SetScriptLanguage(script_lang);
}

lldb::LanguageType SBDebuggerGetREPLLanguage(SBDebuggerRef instance) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return unwrapped->GetREPLLanguage();
}

void SBDebuggerSetREPLLanguage(SBDebuggerRef instance,
                               lldb::LanguageType repl_lang) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  unwrapped->SetREPLLanguage(repl_lang);
}

bool SBDebuggerGetCloseInputOnEOF(SBDebuggerRef instance) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return unwrapped->GetCloseInputOnEOF();
}

void SBDebuggerSetCloseInputOnEOF(SBDebuggerRef instance, bool b) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  unwrapped->SetCloseInputOnEOF(b);
}

SBTypeCategoryRef SBDebuggerGetCategory(SBDebuggerRef instance,
                                        const char *category_name) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return reinterpret_cast<SBTypeCategoryRef>(
      new SBTypeCategory(unwrapped->GetCategory(category_name)));
}

SBTypeCategoryRef SBDebuggerCreateCategory(SBDebuggerRef instance,
                                           const char *category_name) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return reinterpret_cast<SBTypeCategoryRef>(
      new SBTypeCategory(unwrapped->CreateCategory(category_name)));
}

bool SBDebuggerDeleteCategory(SBDebuggerRef instance,
                              const char *category_name) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return unwrapped->DeleteCategory(category_name);
}

uint32_t SBDebuggerGetNumCategories(SBDebuggerRef instance) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return unwrapped->GetNumCategories();
}

SBTypeCategoryRef SBDebuggerGetCategoryAtIndex(SBDebuggerRef instance,
                                               uint32_t var0) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return reinterpret_cast<SBTypeCategoryRef>(
      new SBTypeCategory(unwrapped->GetCategoryAtIndex(var0)));
}

SBTypeCategoryRef SBDebuggerGetDefaultCategory(SBDebuggerRef instance) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return reinterpret_cast<SBTypeCategoryRef>(
      new SBTypeCategory(unwrapped->GetDefaultCategory()));
}

SBTypeFormatRef SBDebuggerGetFormatForType(SBDebuggerRef instance,
                                           SBTypeNameSpecifierRef var0) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return reinterpret_cast<SBTypeFormatRef>(
      new SBTypeFormat(unwrapped->GetFormatForType(
          *reinterpret_cast<SBTypeNameSpecifier *>(var0))));
}

SBTypeSummaryRef SBDebuggerGetSummaryForType(SBDebuggerRef instance,
                                             SBTypeNameSpecifierRef var0) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return reinterpret_cast<SBTypeSummaryRef>(
      new SBTypeSummary(unwrapped->GetSummaryForType(
          *reinterpret_cast<SBTypeNameSpecifier *>(var0))));
}

SBTypeFilterRef SBDebuggerGetFilterForType(SBDebuggerRef instance,
                                           SBTypeNameSpecifierRef var0) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return reinterpret_cast<SBTypeFilterRef>(
      new SBTypeFilter(unwrapped->GetFilterForType(
          *reinterpret_cast<SBTypeNameSpecifier *>(var0))));
}

SBTypeSyntheticRef SBDebuggerGetSyntheticForType(SBDebuggerRef instance,
                                                 SBTypeNameSpecifierRef var0) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return reinterpret_cast<SBTypeSyntheticRef>(
      new SBTypeSynthetic(unwrapped->GetSyntheticForType(
          *reinterpret_cast<SBTypeNameSpecifier *>(var0))));
}

void SBDebuggerRunCommandInterpreter(SBDebuggerRef instance,
                                     bool auto_handle_events,
                                     bool spawn_thread) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  unwrapped->RunCommandInterpreter(auto_handle_events, spawn_thread);
}

void SBDebuggerRunCommandInterpreter2(SBDebuggerRef instance,
                                      bool auto_handle_events,
                                      bool spawn_thread,
                                      SBCommandInterpreterRunOptionsRef options,
                                      int num_errors, bool quit_requested,
                                      bool stopped_for_crash) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  unwrapped->RunCommandInterpreter(
      auto_handle_events, spawn_thread,
      *reinterpret_cast<SBCommandInterpreterRunOptions *>(options), num_errors,
      quit_requested, stopped_for_crash);
}

void SBDebuggerRunCommandInterpreter3(
    SBDebuggerRef instance, SBCommandInterpreterRunOptionsRef options) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  unwrapped->RunCommandInterpreter(
      *reinterpret_cast<SBCommandInterpreterRunOptions *>(options));
}

SBErrorRef SBDebuggerRunREPL(SBDebuggerRef instance,
                             lldb::LanguageType language,
                             const char *repl_options) {
  SBDebugger *unwrapped = reinterpret_cast<SBDebugger *>(instance);
  return reinterpret_cast<SBErrorRef>(
      new SBError(unwrapped->RunREPL(language, repl_options)));
}

#ifdef __cplusplus
}
#endif
