//===-- SBCommandInterpreterBinding.cpp -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/LLDB.h"
#include "lldb/API/SBCommandInterpreterRunOptions.h"
#include "lldb/Bindings/LLDBBinding.h"

using namespace lldb;

#ifdef __cplusplus
extern "C" {
#endif

SBCommandInterpreterRunOptionsRef CreateSBCommandInterpreterRunOptions() {
  return reinterpret_cast<SBCommandInterpreterRunOptionsRef>(
      new SBCommandInterpreterRunOptions());
}

void DisposeSBCommandInterpreterRunOptions(
    SBCommandInterpreterRunOptionsRef instance) {
  delete reinterpret_cast<SBCommandInterpreterRunOptions *>(instance);
}

bool SBCommandInterpreterRunOptionsGetStopOnContinue(
    SBCommandInterpreterRunOptionsRef instance) {
  SBCommandInterpreterRunOptions *unwrapped =
      reinterpret_cast<SBCommandInterpreterRunOptions *>(instance);
  return unwrapped->GetStopOnContinue();
}

void SBCommandInterpreterRunOptionsSetStopOnContinue(
    SBCommandInterpreterRunOptionsRef instance, bool var0) {
  SBCommandInterpreterRunOptions *unwrapped =
      reinterpret_cast<SBCommandInterpreterRunOptions *>(instance);
  unwrapped->SetStopOnContinue(var0);
}

bool SBCommandInterpreterRunOptionsGetStopOnError(
    SBCommandInterpreterRunOptionsRef instance) {
  SBCommandInterpreterRunOptions *unwrapped =
      reinterpret_cast<SBCommandInterpreterRunOptions *>(instance);
  return unwrapped->GetStopOnError();
}

void SBCommandInterpreterRunOptionsSetStopOnError(
    SBCommandInterpreterRunOptionsRef instance, bool var0) {
  SBCommandInterpreterRunOptions *unwrapped =
      reinterpret_cast<SBCommandInterpreterRunOptions *>(instance);
  unwrapped->SetStopOnError(var0);
}

bool SBCommandInterpreterRunOptionsGetStopOnCrash(
    SBCommandInterpreterRunOptionsRef instance) {
  SBCommandInterpreterRunOptions *unwrapped =
      reinterpret_cast<SBCommandInterpreterRunOptions *>(instance);
  return unwrapped->GetStopOnCrash();
}

void SBCommandInterpreterRunOptionsSetStopOnCrash(
    SBCommandInterpreterRunOptionsRef instance, bool var0) {
  SBCommandInterpreterRunOptions *unwrapped =
      reinterpret_cast<SBCommandInterpreterRunOptions *>(instance);
  unwrapped->SetStopOnCrash(var0);
}

bool SBCommandInterpreterRunOptionsGetEchoCommands(
    SBCommandInterpreterRunOptionsRef instance) {
  SBCommandInterpreterRunOptions *unwrapped =
      reinterpret_cast<SBCommandInterpreterRunOptions *>(instance);
  return unwrapped->GetEchoCommands();
}

void SBCommandInterpreterRunOptionsSetEchoCommands(
    SBCommandInterpreterRunOptionsRef instance, bool var0) {
  SBCommandInterpreterRunOptions *unwrapped =
      reinterpret_cast<SBCommandInterpreterRunOptions *>(instance);
  unwrapped->SetEchoCommands(var0);
}

bool SBCommandInterpreterRunOptionsGetEchoCommentCommands(
    SBCommandInterpreterRunOptionsRef instance) {
  SBCommandInterpreterRunOptions *unwrapped =
      reinterpret_cast<SBCommandInterpreterRunOptions *>(instance);
  return unwrapped->GetEchoCommentCommands();
}

void SBCommandInterpreterRunOptionsSetEchoCommentCommands(
    SBCommandInterpreterRunOptionsRef instance, bool echo) {
  SBCommandInterpreterRunOptions *unwrapped =
      reinterpret_cast<SBCommandInterpreterRunOptions *>(instance);
  unwrapped->SetEchoCommentCommands(echo);
}

bool SBCommandInterpreterRunOptionsGetPrintResults(
    SBCommandInterpreterRunOptionsRef instance) {
  SBCommandInterpreterRunOptions *unwrapped =
      reinterpret_cast<SBCommandInterpreterRunOptions *>(instance);
  return unwrapped->GetPrintResults();
}

void SBCommandInterpreterRunOptionsSetPrintResults(
    SBCommandInterpreterRunOptionsRef instance, bool var0) {
  SBCommandInterpreterRunOptions *unwrapped =
      reinterpret_cast<SBCommandInterpreterRunOptions *>(instance);
  unwrapped->SetPrintResults(var0);
}

bool SBCommandInterpreterRunOptionsGetAddToHistory(
    SBCommandInterpreterRunOptionsRef instance) {
  SBCommandInterpreterRunOptions *unwrapped =
      reinterpret_cast<SBCommandInterpreterRunOptions *>(instance);
  return unwrapped->GetAddToHistory();
}

void SBCommandInterpreterRunOptionsSetAddToHistory(
    SBCommandInterpreterRunOptionsRef instance, bool var0) {
  SBCommandInterpreterRunOptions *unwrapped =
      reinterpret_cast<SBCommandInterpreterRunOptions *>(instance);
  unwrapped->SetAddToHistory(var0);
}

SBCommandInterpreterRef
CloneSBCommandInterpreter(SBCommandInterpreterRef instance) {
  return reinterpret_cast<SBCommandInterpreterRef>(new SBCommandInterpreter(
      *reinterpret_cast<SBCommandInterpreter *>(instance)));
}

void DisposeSBCommandInterpreter(SBCommandInterpreterRef instance) {
  delete reinterpret_cast<SBCommandInterpreter *>(instance);
}

const char *SBCommandInterpreterGetArgumentTypeAsCString(
    const lldb::CommandArgumentType arg_type) {
  return lldb::SBCommandInterpreter::GetArgumentTypeAsCString(arg_type);
}

const char *SBCommandInterpreterGetArgumentDescriptionAsCString(
    const lldb::CommandArgumentType arg_type) {
  return lldb::SBCommandInterpreter::GetArgumentDescriptionAsCString(arg_type);
}

bool SBCommandInterpreterEventIsCommandInterpreterEvent(SBEventRef event) {
  return lldb::SBCommandInterpreter::EventIsCommandInterpreterEvent(
      *reinterpret_cast<SBEvent *>(event));
}

bool SBCommandInterpreterIsValid(SBCommandInterpreterRef instance) {
  SBCommandInterpreter *unwrapped =
      reinterpret_cast<SBCommandInterpreter *>(instance);
  return unwrapped->IsValid();
}

bool SBCommandInterpreterCommandExists(SBCommandInterpreterRef instance,
                                       const char *cmd) {
  SBCommandInterpreter *unwrapped =
      reinterpret_cast<SBCommandInterpreter *>(instance);
  return unwrapped->CommandExists(cmd);
}

bool SBCommandInterpreterAliasExists(SBCommandInterpreterRef instance,
                                     const char *cmd) {
  SBCommandInterpreter *unwrapped =
      reinterpret_cast<SBCommandInterpreter *>(instance);
  return unwrapped->AliasExists(cmd);
}

SBBroadcasterRef
SBCommandInterpreterGetBroadcaster(SBCommandInterpreterRef instance) {
  SBCommandInterpreter *unwrapped =
      reinterpret_cast<SBCommandInterpreter *>(instance);
  return reinterpret_cast<SBBroadcasterRef>(
      new SBBroadcaster(unwrapped->GetBroadcaster()));
}

const char *SBCommandInterpreterGetBroadcasterClass() {
  return lldb::SBCommandInterpreter::GetBroadcasterClass();
}

bool SBCommandInterpreterHasCommands(SBCommandInterpreterRef instance) {
  SBCommandInterpreter *unwrapped =
      reinterpret_cast<SBCommandInterpreter *>(instance);
  return unwrapped->HasCommands();
}

bool SBCommandInterpreterHasAliases(SBCommandInterpreterRef instance) {
  SBCommandInterpreter *unwrapped =
      reinterpret_cast<SBCommandInterpreter *>(instance);
  return unwrapped->HasAliases();
}

bool SBCommandInterpreterHasAliasOptions(SBCommandInterpreterRef instance) {
  SBCommandInterpreter *unwrapped =
      reinterpret_cast<SBCommandInterpreter *>(instance);
  return unwrapped->HasAliasOptions();
}

bool SBCommandInterpreterIsInteractive(SBCommandInterpreterRef instance) {
  SBCommandInterpreter *unwrapped =
      reinterpret_cast<SBCommandInterpreter *>(instance);
  return unwrapped->IsInteractive();
}

SBProcessRef SBCommandInterpreterGetProcess(SBCommandInterpreterRef instance) {
  SBCommandInterpreter *unwrapped =
      reinterpret_cast<SBCommandInterpreter *>(instance);
  return reinterpret_cast<SBProcessRef>(new SBProcess(unwrapped->GetProcess()));
}

SBDebuggerRef
SBCommandInterpreterGetDebugger(SBCommandInterpreterRef instance) {
  SBCommandInterpreter *unwrapped =
      reinterpret_cast<SBCommandInterpreter *>(instance);
  return reinterpret_cast<SBDebuggerRef>(
      new SBDebugger(unwrapped->GetDebugger()));
}

SBCommandRef
SBCommandInterpreterAddMultiwordCommand(SBCommandInterpreterRef instance,
                                        const char *name, const char *help) {
  SBCommandInterpreter *unwrapped =
      reinterpret_cast<SBCommandInterpreter *>(instance);
  return reinterpret_cast<SBCommandRef>(
      new SBCommand(unwrapped->AddMultiwordCommand(name, help)));
}

SBCommandRef SBCommandInterpreterAddCommand(SBCommandInterpreterRef instance,
                                            const char *name,
                                            SBCommandPluginInterfaceRef impl,
                                            const char *help) {
  SBCommandInterpreter *unwrapped =
      reinterpret_cast<SBCommandInterpreter *>(instance);
  return reinterpret_cast<SBCommandRef>(new SBCommand(unwrapped->AddCommand(
      name, reinterpret_cast<SBCommandPluginInterface *>(impl), help)));
}

void SBCommandInterpreterSourceInitFileInGlobalDirectory(
    SBCommandInterpreterRef instance, SBCommandReturnObjectRef result) {
  SBCommandInterpreter *unwrapped =
      reinterpret_cast<SBCommandInterpreter *>(instance);
  unwrapped->SourceInitFileInGlobalDirectory(
      *reinterpret_cast<SBCommandReturnObject *>(result));
}

void SBCommandInterpreterSourceInitFileInHomeDirectory(
    SBCommandInterpreterRef instance, SBCommandReturnObjectRef result) {
  SBCommandInterpreter *unwrapped =
      reinterpret_cast<SBCommandInterpreter *>(instance);
  unwrapped->SourceInitFileInHomeDirectory(
      *reinterpret_cast<SBCommandReturnObject *>(result));
}

void SBCommandInterpreterSourceInitFileInHomeDirectory2(
    SBCommandInterpreterRef instance, SBCommandReturnObjectRef result,
    bool is_repl) {
  SBCommandInterpreter *unwrapped =
      reinterpret_cast<SBCommandInterpreter *>(instance);
  unwrapped->SourceInitFileInHomeDirectory(
      *reinterpret_cast<SBCommandReturnObject *>(result), is_repl);
}

void SBCommandInterpreterSourceInitFileInCurrentWorkingDirectory(
    SBCommandInterpreterRef instance, SBCommandReturnObjectRef result) {
  SBCommandInterpreter *unwrapped =
      reinterpret_cast<SBCommandInterpreter *>(instance);
  unwrapped->SourceInitFileInCurrentWorkingDirectory(
      *reinterpret_cast<SBCommandReturnObject *>(result));
}

enum lldb::ReturnStatus SBCommandInterpreterHandleCommand(
    SBCommandInterpreterRef instance, const char *command_line,
    SBCommandReturnObjectRef result, bool add_to_history) {
  SBCommandInterpreter *unwrapped =
      reinterpret_cast<SBCommandInterpreter *>(instance);
  return unwrapped->HandleCommand(
      command_line, *reinterpret_cast<SBCommandReturnObject *>(result),
      add_to_history);
}

enum lldb::ReturnStatus SBCommandInterpreterHandleCommand2(
    SBCommandInterpreterRef instance, const char *command_line,
    SBExecutionContextRef exe_ctx, SBCommandReturnObjectRef result,
    bool add_to_history) {
  SBCommandInterpreter *unwrapped =
      reinterpret_cast<SBCommandInterpreter *>(instance);
  return unwrapped->HandleCommand(
      command_line, *reinterpret_cast<SBExecutionContext *>(exe_ctx),
      *reinterpret_cast<SBCommandReturnObject *>(result), add_to_history);
}

void SBCommandInterpreterHandleCommandsFromFile(
    SBCommandInterpreterRef instance, SBFileSpecRef file,
    SBExecutionContextRef override_context,
    SBCommandInterpreterRunOptionsRef options,
    SBCommandReturnObjectRef result) {
  SBCommandInterpreter *unwrapped =
      reinterpret_cast<SBCommandInterpreter *>(instance);
  unwrapped->HandleCommandsFromFile(
      *reinterpret_cast<SBFileSpec *>(file),
      *reinterpret_cast<SBExecutionContext *>(override_context),
      *reinterpret_cast<SBCommandInterpreterRunOptions *>(options),
      *reinterpret_cast<SBCommandReturnObject *>(result));
}

int SBCommandInterpreterHandleCompletion(
    SBCommandInterpreterRef instance, const char *current_line,
    const char *cursor, const char *last_char, int match_start_point,
    int max_return_elements, SBStringListRef matches) {
  SBCommandInterpreter *unwrapped =
      reinterpret_cast<SBCommandInterpreter *>(instance);
  return unwrapped->HandleCompletion(
      current_line, cursor, last_char, match_start_point, max_return_elements,
      *reinterpret_cast<SBStringList *>(matches));
}

int SBCommandInterpreterHandleCompletion2(SBCommandInterpreterRef instance,
                                          const char *current_line,
                                          uint32_t cursor_pos,
                                          int match_start_point,
                                          int max_return_elements,
                                          SBStringListRef matches) {
  SBCommandInterpreter *unwrapped =
      reinterpret_cast<SBCommandInterpreter *>(instance);
  return unwrapped->HandleCompletion(
      current_line, cursor_pos, match_start_point, max_return_elements,
      *reinterpret_cast<SBStringList *>(matches));
}

int SBCommandInterpreterHandleCompletionWithDescriptions(
    SBCommandInterpreterRef instance, const char *current_line,
    const char *cursor, const char *last_char, int match_start_point,
    int max_return_elements, SBStringListRef matches,
    SBStringListRef descriptions) {
  SBCommandInterpreter *unwrapped =
      reinterpret_cast<SBCommandInterpreter *>(instance);
  return unwrapped->HandleCompletionWithDescriptions(
      current_line, cursor, last_char, match_start_point, max_return_elements,
      *reinterpret_cast<SBStringList *>(matches),
      *reinterpret_cast<SBStringList *>(descriptions));
}

int SBCommandInterpreterHandleCompletionWithDescriptions2(
    SBCommandInterpreterRef instance, const char *current_line,
    uint32_t cursor_pos, int match_start_point, int max_return_elements,
    SBStringListRef matches, SBStringListRef descriptions) {
  SBCommandInterpreter *unwrapped =
      reinterpret_cast<SBCommandInterpreter *>(instance);
  return unwrapped->HandleCompletionWithDescriptions(
      current_line, cursor_pos, match_start_point, max_return_elements,
      *reinterpret_cast<SBStringList *>(matches),
      *reinterpret_cast<SBStringList *>(descriptions));
}

bool SBCommandInterpreterWasInterrupted(SBCommandInterpreterRef instance) {
  SBCommandInterpreter *unwrapped =
      reinterpret_cast<SBCommandInterpreter *>(instance);
  return unwrapped->WasInterrupted();
}

bool SBCommandInterpreterSetCommandOverrideCallback(
    SBCommandInterpreterRef instance, const char *command_name,
    lldb::CommandOverrideCallback callback, void *baton) {
  SBCommandInterpreter *unwrapped =
      reinterpret_cast<SBCommandInterpreter *>(instance);
  return unwrapped->SetCommandOverrideCallback(command_name, callback, baton);
}

bool SBCommandInterpreterIsActive(SBCommandInterpreterRef instance) {
  SBCommandInterpreter *unwrapped =
      reinterpret_cast<SBCommandInterpreter *>(instance);
  return unwrapped->IsActive();
}

const char *SBCommandInterpreterGetIOHandlerControlSequence(
    SBCommandInterpreterRef instance, char ch) {
  SBCommandInterpreter *unwrapped =
      reinterpret_cast<SBCommandInterpreter *>(instance);
  return unwrapped->GetIOHandlerControlSequence(ch);
}

bool SBCommandInterpreterGetPromptOnQuit(SBCommandInterpreterRef instance) {
  SBCommandInterpreter *unwrapped =
      reinterpret_cast<SBCommandInterpreter *>(instance);
  return unwrapped->GetPromptOnQuit();
}

void SBCommandInterpreterSetPromptOnQuit(SBCommandInterpreterRef instance,
                                         bool b) {
  SBCommandInterpreter *unwrapped =
      reinterpret_cast<SBCommandInterpreter *>(instance);
  unwrapped->SetPromptOnQuit(b);
}

bool SBCommandPluginInterfaceDoExecute(SBCommandPluginInterfaceRef instance,
                                       SBDebuggerRef, char **var0,
                                       SBCommandReturnObjectRef var1) {
  SBCommandPluginInterface *unwrapped =
      reinterpret_cast<SBCommandPluginInterface *>(instance);
  return unwrapped->DoExecute(*reinterpret_cast<SBDebugger *>(var1), var0,
                              *reinterpret_cast<SBCommandReturnObject *>(var1));
}

SBCommandPluginInterfaceRef
CloneSBCommandPluginInterface(SBCommandPluginInterfaceRef instance) {
  return reinterpret_cast<SBCommandPluginInterfaceRef>(
      new SBCommandPluginInterface(
          *reinterpret_cast<SBCommandPluginInterface *>(instance)));
}

void DisposeSBCommandPluginInterface(SBCommandPluginInterfaceRef instance) {
  delete reinterpret_cast<SBCommandPluginInterface *>(instance);
}

SBCommandRef CreateSBCommand() {
  return reinterpret_cast<SBCommandRef>(new SBCommand());
}

bool SBCommandIsValid(SBCommandRef instance) {
  SBCommand *unwrapped = reinterpret_cast<SBCommand *>(instance);
  return unwrapped->IsValid();
}

const char *SBCommandGetName(SBCommandRef instance) {
  SBCommand *unwrapped = reinterpret_cast<SBCommand *>(instance);
  return unwrapped->GetName();
}

const char *SBCommandGetHelp(SBCommandRef instance) {
  SBCommand *unwrapped = reinterpret_cast<SBCommand *>(instance);
  return unwrapped->GetHelp();
}

const char *SBCommandGetHelpLong(SBCommandRef instance) {
  SBCommand *unwrapped = reinterpret_cast<SBCommand *>(instance);
  return unwrapped->GetHelpLong();
}

void SBCommandSetHelp(SBCommandRef instance, const char *var0) {
  SBCommand *unwrapped = reinterpret_cast<SBCommand *>(instance);
  unwrapped->SetHelp(var0);
}

void SBCommandSetHelpLong(SBCommandRef instance, const char *var0) {
  SBCommand *unwrapped = reinterpret_cast<SBCommand *>(instance);
  unwrapped->SetHelpLong(var0);
}

SBCommandRef SBCommandAddMultiwordCommand(SBCommandRef instance,
                                          const char *name, const char *help) {
  SBCommand *unwrapped = reinterpret_cast<SBCommand *>(instance);
  return reinterpret_cast<SBCommandRef>(
      new SBCommand(unwrapped->AddMultiwordCommand(name, help)));
}

SBCommandRef SBCommandAddCommand(SBCommandRef instance, const char *name,
                                 SBCommandPluginInterfaceRef impl,
                                 const char *help) {
  SBCommand *unwrapped = reinterpret_cast<SBCommand *>(instance);
  return reinterpret_cast<SBCommandRef>(new SBCommand(unwrapped->AddCommand(
      name, reinterpret_cast<SBCommandPluginInterface *>(impl), help)));
}

SBCommandRef CloneSBCommand(SBCommandRef instance) {
  return reinterpret_cast<SBCommandRef>(
      new SBCommand(*reinterpret_cast<SBCommand *>(instance)));
}

void DisposeSBCommand(SBCommandRef instance) {
  delete reinterpret_cast<SBCommand *>(instance);
}

#ifdef __cplusplus
}
#endif
