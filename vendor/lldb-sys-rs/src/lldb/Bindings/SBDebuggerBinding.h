//===-- SBDebuggerBinding.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBDebuggerBinding_h_
#define LLDB_SBDebuggerBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"
#ifndef BINDGEN
#include <lldb/API/SBDebugger.h>
#endif

#ifndef BINDGEN
typedef unsigned long (*Callback)(void *baton, lldb::SBInputReader *reader,
                                  lldb::InputReaderAction notification,
                                  const char *bytes, unsigned long bytes_len);
#endif // BINDGEN

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBInputReaderRef CreateSBInputReader(void);

LLDB_API SBInputReaderRef CloneSBInputReader(SBInputReaderRef instance);

LLDB_API void DisposeSBInputReader(SBInputReaderRef instance);

#ifndef BINDGEN
LLDB_API SBErrorRef SBInputReaderInitialize(SBInputReaderRef instance,
                                            SBDebuggerRef, Callback callback,
                                            void *callback_baton,
                                            ENUM(InputReaderGranularity),
                                            const char *, const char *, bool);
#endif // BINDGEN

LLDB_API void SBInputReaderSetIsDone(SBInputReaderRef instance, bool);

LLDB_API bool SBInputReaderIsActive(SBInputReaderRef instance);

LLDB_API const char *SBDebuggerGetBroadcasterClass(void);

LLDB_API SBBroadcasterRef SBDebuggerGetBroadcaster(SBDebuggerRef instance);

LLDB_API const char *SBDebuggerGetProgressFromEvent(SBEventRef event,
                                                    uint64_t *progress_id,
                                                    uint64_t *completed,
                                                    uint64_t *total,
                                                    bool *is_debugger_specific);

LLDB_API void SBDebuggerInitialize(void);

LLDB_API void SBDebuggerTerminate(void);

LLDB_API SBDebuggerRef SBDebuggerCreate(void);

LLDB_API SBDebuggerRef SBDebuggerCreate2(bool source_init_files);

#ifndef BINDGEN
LLDB_API SBDebuggerRef SBDebuggerCreate3(bool source_init_files,
                                         lldb::LogOutputCallback callback,
                                         void *baton);
#endif // BINDGEN

LLDB_API void SBDebuggerDestroy(SBDebuggerRef debugger);

LLDB_API void SBDebuggerMemoryPressureDetected(void);

LLDB_API SBDebuggerRef CreateSBDebugger(void);

LLDB_API SBDebuggerRef CloneSBDebugger(SBDebuggerRef instance);

LLDB_API void DisposeSBDebugger(SBDebuggerRef instance);

LLDB_API bool SBDebuggerIsValid(SBDebuggerRef instance);

LLDB_API void SBDebuggerClear(SBDebuggerRef instance);

LLDB_API void SBDebuggerSetAsync(SBDebuggerRef instance, bool b);

LLDB_API bool SBDebuggerGetAsync(SBDebuggerRef instance);

LLDB_API void SBDebuggerSkipLLDBInitFiles(SBDebuggerRef instance, bool b);

LLDB_API void SBDebuggerSkipAppInitFiles(SBDebuggerRef instance, bool b);

LLDB_API SBErrorRef SBDebuggerSetInputString(SBDebuggerRef instance,
                                             const char *data);

LLDB_API SBErrorRef SBDebuggerSetInputFile(SBDebuggerRef instance,
                                           SBFileRef file);

LLDB_API SBErrorRef SBDebuggerSetOutputFile(SBDebuggerRef instance,
                                            SBFileRef file);

LLDB_API SBErrorRef SBDebuggerSetErrorFile(SBDebuggerRef instance,
                                           SBFileRef file);

LLDB_API SBFileRef SBDebuggerGetInputFile(SBDebuggerRef instance);

LLDB_API SBFileRef SBDebuggerGetOutputFile(SBDebuggerRef instance);

LLDB_API SBFileRef SBDebuggerGetErrorFile(SBDebuggerRef instance);

LLDB_API void SBDebuggerSaveInputTerminalState(SBDebuggerRef instance);

LLDB_API void SBDebuggerRestoreInputTerminalState(SBDebuggerRef instance);

LLDB_API SBCommandInterpreterRef
SBDebuggerGetCommandInterpreter(SBDebuggerRef instance);

LLDB_API void SBDebuggerHandleCommand(SBDebuggerRef instance,
                                      const char *command);

LLDB_API SBListenerRef SBDebuggerGetListener(SBDebuggerRef instance);

LLDB_API void SBDebuggerHandleProcessEvent(SBDebuggerRef instance,
                                           SBProcessRef process,
                                           SBEventRef event, SBFileRef out,
                                           SBFileRef err);

LLDB_API SBTargetRef SBDebuggerCreateTarget(
    SBDebuggerRef instance, const char *filename, const char *target_triple,
    const char *platform_name, bool add_dependent_modules, SBErrorRef error);

LLDB_API SBTargetRef SBDebuggerCreateTargetWithFileAndTargetTriple(
    SBDebuggerRef instance, const char *filename, const char *target_triple);

LLDB_API SBTargetRef SBDebuggerCreateTargetWithFileAndArch(
    SBDebuggerRef instance, const char *filename, const char *archname);

LLDB_API SBTargetRef SBDebuggerCreateTarget2(SBDebuggerRef instance,
                                             const char *filename);

LLDB_API SBTargetRef SBDebuggerGetDummyTarget(SBDebuggerRef instance);

LLDB_API bool SBDebuggerDeleteTarget(SBDebuggerRef instance,
                                     SBTargetRef target);

LLDB_API SBTargetRef SBDebuggerGetTargetAtIndex(SBDebuggerRef instance,
                                                uint32_t idx);

LLDB_API uint32_t SBDebuggerGetIndexOfTarget(SBDebuggerRef instance,
                                             SBTargetRef target);

LLDB_API SBTargetRef SBDebuggerFindTargetWithProcessID(SBDebuggerRef instance,
                                                       lldb_pid_t pid);

LLDB_API SBTargetRef SBDebuggerFindTargetWithFileAndArch(SBDebuggerRef instance,
                                                         const char *filename,
                                                         const char *arch);

LLDB_API uint32_t SBDebuggerGetNumTargets(SBDebuggerRef instance);

LLDB_API SBTargetRef SBDebuggerGetSelectedTarget(SBDebuggerRef instance);

LLDB_API void SBDebuggerSetSelectedTarget(SBDebuggerRef instance,
                                          SBTargetRef target);

LLDB_API SBPlatformRef SBDebuggerGetSelectedPlatform(SBDebuggerRef instance);

LLDB_API void SBDebuggerSetSelectedPlatform(SBDebuggerRef instance,
                                            SBPlatformRef platform);

LLDB_API uint32_t SBDebuggerGetNumPlatforms(SBDebuggerRef instance);

LLDB_API SBPlatformRef SBDebuggerGetPlatformAtIndex(SBDebuggerRef instance,
                                                    uint32_t idx);

LLDB_API uint32_t SBDebuggerGetNumAvailablePlatforms(SBDebuggerRef instance);

LLDB_API SBStructuredDataRef
SBDebuggerGetAvailablePlatformInfoAtIndex(SBDebuggerRef instance, uint32_t idx);

LLDB_API SBSourceManagerRef SBDebuggerGetSourceManager(SBDebuggerRef instance);

LLDB_API SBErrorRef SBDebuggerSetCurrentPlatform(SBDebuggerRef instance,
                                                 const char *platform_name);

LLDB_API bool SBDebuggerSetCurrentPlatformSDKRoot(SBDebuggerRef instance,
                                                  const char *sysroot);

LLDB_API bool SBDebuggerSetUseExternalEditor(SBDebuggerRef instance,
                                             bool input);

LLDB_API bool SBDebuggerGetUseExternalEditor(SBDebuggerRef instance);

LLDB_API bool SBDebuggerSetUseColor(SBDebuggerRef instance, bool use_color);

LLDB_API bool SBDebuggerGetUseColor(SBDebuggerRef instance);

LLDB_API bool SBDebuggerSetUseSourceCache(SBDebuggerRef instance,
                                          bool use_source_cache);

LLDB_API bool SBDebuggerGetUseSourceCache(SBDebuggerRef instance);

LLDB_API bool SBDebuggerGetDefaultArchitecture(char *arch_name,
                                               size_t arch_name_len);

LLDB_API bool SBDebuggerSetDefaultArchitecture(const char *arch_name);

LLDB_API ENUM(ScriptLanguage)
    SBDebuggerGetScriptingLanguage(SBDebuggerRef instance,
                                   const char *script_language_name);

LLDB_API SBStructuredDataRef SBDebuggerGetScriptInterpreterInfo(
    SBDebuggerRef instance, lldb::ScriptLanguage language);

LLDB_API const char *SBDebuggerGetVersionString(void);

LLDB_API const char *SBDebuggerStateAsCString(ENUM(StateType) state);

LLDB_API bool SBDebuggerStateIsRunningState(ENUM(StateType) state);

LLDB_API bool SBDebuggerStateIsStoppedState(ENUM(StateType) state);

LLDB_API bool SBDebuggerEnableLog(SBDebuggerRef instance, const char *channel,
                                  const char **categories);

#ifndef BINDGEN
LLDB_API void SBDebuggerSetLoggingCallback(SBDebuggerRef instance,
                                           lldb::LogOutputCallback log_callback,
                                           void *baton);
#endif // BINDGEN

LLDB_API void SBDebuggerDispatchInput(SBDebuggerRef instance, void *baton,
                                      const void *data, size_t data_len);

LLDB_API void SBDebuggerDispatchInput2(SBDebuggerRef instance, const void *data,
                                       size_t data_len);

LLDB_API void SBDebuggerDispatchInputInterrupt(SBDebuggerRef instance);

LLDB_API void SBDebuggerDispatchInputEndOfFile(SBDebuggerRef instance);

LLDB_API void SBDebuggerPushInputReader(SBDebuggerRef instance,
                                        SBInputReaderRef reader);

LLDB_API const char *SBDebuggerGetInstanceName(SBDebuggerRef instance);

LLDB_API SBDebuggerRef SBDebuggerFindDebuggerWithID(int id);

LLDB_API SBErrorRef
SBDebuggerSetInternalVariable(const char *var_name, const char *value,
                              const char *debugger_instance_name);

LLDB_API SBStringListRef SBDebuggerGetInternalVariableValue(
    const char *var_name, const char *debugger_instance_name);

LLDB_API bool SBDebuggerGetDescription(SBDebuggerRef instance,
                                       SBStreamRef description);

LLDB_API uint32_t SBDebuggerGetTerminalWidth(SBDebuggerRef instance);

LLDB_API void SBDebuggerSetTerminalWidth(SBDebuggerRef instance,
                                         uint32_t term_width);

LLDB_API lldb_user_id_t SBDebuggerGetID(SBDebuggerRef instance);

LLDB_API const char *SBDebuggerGetPrompt(SBDebuggerRef instance);

LLDB_API void SBDebuggerSetPrompt(SBDebuggerRef instance, const char *prompt);

LLDB_API ENUM(ScriptLanguage)
    SBDebuggerGetScriptLanguage(SBDebuggerRef instance);

LLDB_API void SBDebuggerSetScriptLanguage(SBDebuggerRef instance,
                                          ENUM(ScriptLanguage) script_lang);

LLDB_API lldb::LanguageType SBDebuggerGetREPLLanguage(SBDebuggerRef instance);

LLDB_API void SBDebuggerSetREPLLanguage(SBDebuggerRef instance,
                                        lldb::LanguageType repl_lang);

LLDB_API bool SBDebuggerGetCloseInputOnEOF(SBDebuggerRef instance);

LLDB_API void SBDebuggerSetCloseInputOnEOF(SBDebuggerRef instance, bool b);

LLDB_API SBTypeCategoryRef SBDebuggerGetCategory(SBDebuggerRef instance,
                                                 const char *category_name);

LLDB_API SBTypeCategoryRef SBDebuggerCreateCategory(SBDebuggerRef instance,
                                                    const char *category_name);

LLDB_API bool SBDebuggerDeleteCategory(SBDebuggerRef instance,
                                       const char *category_name);

LLDB_API uint32_t SBDebuggerGetNumCategories(SBDebuggerRef instance);

LLDB_API SBTypeCategoryRef SBDebuggerGetCategoryAtIndex(SBDebuggerRef instance,
                                                        uint32_t);

LLDB_API SBTypeCategoryRef SBDebuggerGetDefaultCategory(SBDebuggerRef instance);

LLDB_API SBTypeFormatRef SBDebuggerGetFormatForType(SBDebuggerRef instance,
                                                    SBTypeNameSpecifierRef);

LLDB_API SBTypeSummaryRef SBDebuggerGetSummaryForType(SBDebuggerRef instance,
                                                      SBTypeNameSpecifierRef);

LLDB_API SBTypeFilterRef SBDebuggerGetFilterForType(SBDebuggerRef instance,
                                                    SBTypeNameSpecifierRef);

LLDB_API SBTypeSyntheticRef
SBDebuggerGetSyntheticForType(SBDebuggerRef instance, SBTypeNameSpecifierRef);

LLDB_API void SBDebuggerRunCommandInterpreter(SBDebuggerRef instance,
                                              bool auto_handle_events,
                                              bool spawn_thread);

LLDB_API void SBDebuggerRunCommandInterpreter2(
    SBDebuggerRef instance, bool auto_handle_events, bool spawn_thread,
    SBCommandInterpreterRunOptionsRef options, int num_errors,
    bool quit_requested, bool stopped_for_crash);

LLDB_API void
SBDebuggerRunCommandInterpreter3(SBDebuggerRef instance,
                                 SBCommandInterpreterRunOptionsRef options);

LLDB_API SBErrorRef SBDebuggerRunREPL(SBDebuggerRef instance,
                                      lldb::LanguageType language,
                                      const char *repl_options);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBDebuggerBinding_h_
