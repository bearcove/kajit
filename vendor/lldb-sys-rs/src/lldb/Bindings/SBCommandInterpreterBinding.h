//===-- SBCommandInterpreterBinding.h ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBCommandInterpreterBinding_h_
#define LLDB_SBCommandInterpreterBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBCommandInterpreterRunOptionsRef
CreateSBCommandInterpreterRunOptions(void);

LLDB_API void DisposeSBCommandInterpreterRunOptions(
    SBCommandInterpreterRunOptionsRef instance);

LLDB_API bool SBCommandInterpreterRunOptionsGetStopOnContinue(
    SBCommandInterpreterRunOptionsRef instance);

LLDB_API void SBCommandInterpreterRunOptionsSetStopOnContinue(
    SBCommandInterpreterRunOptionsRef instance, bool);

LLDB_API bool SBCommandInterpreterRunOptionsGetStopOnError(
    SBCommandInterpreterRunOptionsRef instance);

LLDB_API void SBCommandInterpreterRunOptionsSetStopOnError(
    SBCommandInterpreterRunOptionsRef instance, bool);

LLDB_API bool SBCommandInterpreterRunOptionsGetStopOnCrash(
    SBCommandInterpreterRunOptionsRef instance);

LLDB_API void SBCommandInterpreterRunOptionsSetStopOnCrash(
    SBCommandInterpreterRunOptionsRef instance, bool);

LLDB_API bool SBCommandInterpreterRunOptionsGetEchoCommands(
    SBCommandInterpreterRunOptionsRef instance);

LLDB_API void SBCommandInterpreterRunOptionsSetEchoCommands(
    SBCommandInterpreterRunOptionsRef instance, bool);

LLDB_API bool SBCommandInterpreterRunOptionsGetEchoCommentCommands(
    SBCommandInterpreterRunOptionsRef instance);

LLDB_API void SBCommandInterpreterRunOptionsSetEchoCommentCommands(
    SBCommandInterpreterRunOptionsRef instance, bool echo);

LLDB_API bool SBCommandInterpreterRunOptionsGetPrintResults(
    SBCommandInterpreterRunOptionsRef instance);

LLDB_API void SBCommandInterpreterRunOptionsSetPrintResults(
    SBCommandInterpreterRunOptionsRef instance, bool);

LLDB_API bool SBCommandInterpreterRunOptionsGetAddToHistory(
    SBCommandInterpreterRunOptionsRef instance);

LLDB_API void SBCommandInterpreterRunOptionsSetAddToHistory(
    SBCommandInterpreterRunOptionsRef instance, bool);

LLDB_API SBCommandInterpreterRunOptionsRef
    CreateSBCommandInterpreterRunOptions2(SBCommandInterpreterRunOptionsRef);

LLDB_API SBCommandInterpreterRef
CloneSBCommandInterpreter(SBCommandInterpreterRef instance);

LLDB_API void DisposeSBCommandInterpreter(SBCommandInterpreterRef instance);

LLDB_API const char *
    SBCommandInterpreterGetArgumentTypeAsCString(ENUM(CommandArgumentType)
                                                     arg_type);

LLDB_API const char *SBCommandInterpreterGetArgumentDescriptionAsCString(
    ENUM(CommandArgumentType) arg_type);

LLDB_API bool
SBCommandInterpreterEventIsCommandInterpreterEvent(SBEventRef event);

LLDB_API bool SBCommandInterpreterIsValid(SBCommandInterpreterRef instance);

LLDB_API bool
SBCommandInterpreterCommandExists(SBCommandInterpreterRef instance,
                                  const char *cmd);

LLDB_API bool SBCommandInterpreterAliasExists(SBCommandInterpreterRef instance,
                                              const char *cmd);

LLDB_API SBBroadcasterRef
SBCommandInterpreterGetBroadcaster(SBCommandInterpreterRef instance);

LLDB_API const char *SBCommandInterpreterGetBroadcasterClass(void);

LLDB_API bool SBCommandInterpreterHasCommands(SBCommandInterpreterRef instance);

LLDB_API bool SBCommandInterpreterHasAliases(SBCommandInterpreterRef instance);

LLDB_API bool
SBCommandInterpreterHasAliasOptions(SBCommandInterpreterRef instance);

LLDB_API bool
SBCommandInterpreterIsInteractive(SBCommandInterpreterRef instance);

LLDB_API SBProcessRef
SBCommandInterpreterGetProcess(SBCommandInterpreterRef instance);

LLDB_API SBDebuggerRef
SBCommandInterpreterGetDebugger(SBCommandInterpreterRef instance);

LLDB_API SBCommandRef SBCommandInterpreterAddMultiwordCommand(
    SBCommandInterpreterRef instance, const char *name, const char *help);

LLDB_API SBCommandRef SBCommandInterpreterAddCommand(
    SBCommandInterpreterRef instance, const char *name,
    SBCommandPluginInterfaceRef impl, const char *help);

LLDB_API void SBCommandInterpreterSourceInitFileInGlobalDirectory(
    SBCommandInterpreterRef instance, SBCommandReturnObjectRef result);

LLDB_API void SBCommandInterpreterSourceInitFileInHomeDirectory(
    SBCommandInterpreterRef instance, SBCommandReturnObjectRef result);

LLDB_API void SBCommandInterpreterSourceInitFileInHomeDirectory2(
    SBCommandInterpreterRef instance, SBCommandReturnObjectRef result,
    bool is_repl);

LLDB_API void SBCommandInterpreterSourceInitFileInCurrentWorkingDirectory(
    SBCommandInterpreterRef instance, SBCommandReturnObjectRef result);

LLDB_API ENUM(ReturnStatus)
    SBCommandInterpreterHandleCommand(SBCommandInterpreterRef instance,
                                      const char *command_line,
                                      SBCommandReturnObjectRef result,
                                      bool add_to_history);

LLDB_API ENUM(ReturnStatus)
    SBCommandInterpreterHandleCommand2(SBCommandInterpreterRef instance,
                                       const char *command_line,
                                       SBExecutionContextRef exe_ctx,
                                       SBCommandReturnObjectRef result,
                                       bool add_to_history);

LLDB_API void SBCommandInterpreterHandleCommandsFromFile(
    SBCommandInterpreterRef instance, SBFileSpecRef file,
    SBExecutionContextRef override_context,
    SBCommandInterpreterRunOptionsRef options, SBCommandReturnObjectRef result);

LLDB_API int SBCommandInterpreterHandleCompletion(
    SBCommandInterpreterRef instance, const char *current_line,
    const char *cursor, const char *last_char, int match_start_point,
    int max_return_elements, SBStringListRef matches);

LLDB_API int SBCommandInterpreterHandleCompletion2(
    SBCommandInterpreterRef instance, const char *current_line,
    uint32_t cursor_pos, int match_start_point, int max_return_elements,
    SBStringListRef matches);

LLDB_API int SBCommandInterpreterHandleCompletionWithDescriptions(
    SBCommandInterpreterRef instance, const char *current_line,
    const char *cursor, const char *last_char, int match_start_point,
    int max_return_elements, SBStringListRef matches,
    SBStringListRef descriptions);

LLDB_API int SBCommandInterpreterHandleCompletionWithDescriptions2(
    SBCommandInterpreterRef instance, const char *current_line,
    uint32_t cursor_pos, int match_start_point, int max_return_elements,
    SBStringListRef matches, SBStringListRef descriptions);

LLDB_API bool
SBCommandInterpreterWasInterrupted(SBCommandInterpreterRef instance);

#ifndef BINDGEN
LLDB_API bool SBCommandInterpreterSetCommandOverrideCallback(
    SBCommandInterpreterRef instance, const char *command_name,
    lldb::CommandOverrideCallback callback, void *baton);
#endif // BINDGEN

LLDB_API bool SBCommandInterpreterIsActive(SBCommandInterpreterRef instance);

LLDB_API const char *SBCommandInterpreterGetIOHandlerControlSequence(
    SBCommandInterpreterRef instance, char ch);

LLDB_API bool
SBCommandInterpreterGetPromptOnQuit(SBCommandInterpreterRef instance);

LLDB_API void
SBCommandInterpreterSetPromptOnQuit(SBCommandInterpreterRef instance, bool b);

LLDB_API bool
SBCommandPluginInterfaceDoExecute(SBCommandPluginInterfaceRef instance,
                                  SBDebuggerRef, char **,
                                  SBCommandReturnObjectRef);

LLDB_API SBCommandPluginInterfaceRef
CloneSBCommandPluginInterface(SBCommandPluginInterfaceRef instance);

LLDB_API void
DisposeSBCommandPluginInterface(SBCommandPluginInterfaceRef instance);

LLDB_API SBCommandRef CreateSBCommand(void);

LLDB_API bool SBCommandIsValid(SBCommandRef instance);

LLDB_API const char *SBCommandGetName(SBCommandRef instance);

LLDB_API const char *SBCommandGetHelp(SBCommandRef instance);

LLDB_API const char *SBCommandGetHelpLong(SBCommandRef instance);

LLDB_API void SBCommandSetHelp(SBCommandRef instance, const char *);

LLDB_API void SBCommandSetHelpLong(SBCommandRef instance, const char *);

LLDB_API SBCommandRef SBCommandAddMultiwordCommand(SBCommandRef instance,
                                                   const char *name,
                                                   const char *help);

LLDB_API SBCommandRef SBCommandAddCommand(SBCommandRef instance,
                                          const char *name,
                                          SBCommandPluginInterfaceRef impl,
                                          const char *help);

LLDB_API SBCommandRef CloneSBCommand(SBCommandRef instance);

LLDB_API void DisposeSBCommand(SBCommandRef instance);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBCommandInterpreterBinding_h_
