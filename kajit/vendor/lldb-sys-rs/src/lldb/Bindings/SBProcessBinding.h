//===-- SBProcessBinding.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBProcessBinding_h_
#define LLDB_SBProcessBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBProcessRef CreateSBProcess(void);

LLDB_API SBProcessRef CloneSBProcess(SBProcessRef instance);

LLDB_API void DisposeSBProcess(SBProcessRef instance);

LLDB_API const char *SBProcessGetBroadcasterClassName(void);

LLDB_API const char *SBProcessGetPluginName(SBProcessRef instance);

LLDB_API const char *SBProcessGetShortPluginName(SBProcessRef instance);

LLDB_API void SBProcessClear(SBProcessRef instance);

LLDB_API bool SBProcessIsValid(SBProcessRef instance);

LLDB_API SBTargetRef SBProcessGetTarget(SBProcessRef instance);

LLDB_API ENUM(ByteOrder) SBProcessGetByteOrder(SBProcessRef instance);

LLDB_API size_t SBProcessPutSTDIN(SBProcessRef instance, const char *src,
                                  size_t src_len);

LLDB_API size_t SBProcessGetSTDOUT(SBProcessRef instance, char *dst,
                                   size_t dst_len);

LLDB_API size_t SBProcessGetSTDERR(SBProcessRef instance, char *dst,
                                   size_t dst_len);

LLDB_API size_t SBProcessGetAsyncProfileData(SBProcessRef instance, char *dst,
                                             size_t dst_len);

LLDB_API void SBProcessReportEventState(SBProcessRef instance, SBEventRef event,
                                        SBFileRef out);

LLDB_API void SBProcessAppendEventStateReport(SBProcessRef instance,
                                              SBEventRef event,
                                              SBCommandReturnObjectRef result);

LLDB_API bool SBProcessRemoteAttachToProcessWithID(SBProcessRef instance,
                                                   lldb_pid_t pid,
                                                   SBErrorRef error);

LLDB_API bool SBProcessRemoteLaunch(SBProcessRef instance, const char **argv,
                                    const char **envp, const char *stdin_path,
                                    const char *stdout_path,
                                    const char *stderr_path,
                                    const char *working_directory,
                                    uint32_t launch_flags, bool stop_at_entry,
                                    SBErrorRef error);

LLDB_API uint32_t SBProcessGetNumThreads(SBProcessRef instance);

LLDB_API SBThreadRef SBProcessGetThreadAtIndex(SBProcessRef instance,
                                               size_t index);

LLDB_API SBThreadRef SBProcessGetThreadByID(SBProcessRef instance,
                                            lldb_tid_t sb_thread_id);

LLDB_API SBThreadRef SBProcessGetThreadByIndexID(SBProcessRef instance,
                                                 uint32_t index_id);

LLDB_API SBThreadRef SBProcessGetSelectedThread(SBProcessRef instance);

LLDB_API SBThreadRef SBProcessCreateOSPluginThread(SBProcessRef instance,
                                                   lldb_tid_t tid,
                                                   lldb_addr_t context);

LLDB_API bool SBProcessSetSelectedThread(SBProcessRef instance,
                                         SBThreadRef thread);

LLDB_API bool SBProcessSetSelectedThreadByID(SBProcessRef instance,
                                             lldb_tid_t tid);

LLDB_API bool SBProcessSetSelectedThreadByIndexID(SBProcessRef instance,
                                                  uint32_t index_id);

LLDB_API uint32_t SBProcessGetNumQueues(SBProcessRef instance);

LLDB_API SBQueueRef SBProcessGetQueueAtIndex(SBProcessRef instance,
                                             size_t index);

LLDB_API ENUM(StateType) SBProcessGetState(SBProcessRef instance);

LLDB_API int SBProcessGetExitStatus(SBProcessRef instance);

LLDB_API const char *SBProcessGetExitDescription(SBProcessRef instance);

LLDB_API lldb_pid_t SBProcessGetProcessID(SBProcessRef instance);

LLDB_API uint32_t SBProcessGetUniqueID(SBProcessRef instance);

LLDB_API uint32_t SBProcessGetAddressByteSize(SBProcessRef instance);

LLDB_API SBErrorRef SBProcessDestroy(SBProcessRef instance);

LLDB_API SBErrorRef SBProcessContinue(SBProcessRef instance);

LLDB_API SBErrorRef SBProcessStop(SBProcessRef instance);

LLDB_API SBErrorRef SBProcessKill(SBProcessRef instance);

LLDB_API SBErrorRef SBProcessDetach(SBProcessRef instance);

LLDB_API SBErrorRef SBProcessDetach2(SBProcessRef instance, bool keep_stopped);

LLDB_API SBErrorRef SBProcessSignal(SBProcessRef instance, int signal);

LLDB_API SBUnixSignalsRef SBProcessGetUnixSignals(SBProcessRef instance);

LLDB_API void SBProcessSendAsyncInterrupt(SBProcessRef instance);

LLDB_API uint32_t SBProcessGetStopID(SBProcessRef instance,
                                     bool include_expression_stops);

LLDB_API size_t SBProcessReadMemory(SBProcessRef instance, lldb_addr_t addr,
                                    void *buf, size_t size, SBErrorRef error);

LLDB_API size_t SBProcessWriteMemory(SBProcessRef instance, lldb_addr_t addr,
                                     void *buf, size_t size, SBErrorRef error);

LLDB_API size_t SBProcessReadCStringFromMemory(SBProcessRef instance,
                                               lldb_addr_t addr, void *buf,
                                               size_t size, SBErrorRef error);

LLDB_API uint64_t SBProcessReadUnsignedFromMemory(SBProcessRef instance,
                                                  lldb_addr_t addr,
                                                  uint32_t byte_size,
                                                  SBErrorRef error);

LLDB_API lldb_addr_t SBProcessReadPointerFromMemory(SBProcessRef instance,
                                                    lldb_addr_t addr,
                                                    SBErrorRef error);

LLDB_API ENUM(StateType) SBProcessGetStateFromEvent(SBEventRef event);

LLDB_API bool SBProcessGetRestartedFromEvent(SBEventRef event);

LLDB_API size_t SBProcessGetNumRestartedReasonsFromEvent(SBEventRef event);

LLDB_API const char *
SBProcessGetRestartedReasonAtIndexFromEvent(SBEventRef event, size_t idx);

LLDB_API SBProcessRef SBProcessGetProcessFromEvent(SBEventRef event);

LLDB_API bool SBProcessGetInterruptedFromEvent(SBEventRef event);

LLDB_API SBStructuredDataRef
SBProcessGetStructuredDataFromEvent(SBEventRef event);

LLDB_API bool SBProcessEventIsProcessEvent(SBEventRef event);

LLDB_API bool SBProcessEventIsStructuredDataEvent(SBEventRef event);

LLDB_API SBBroadcasterRef SBProcessGetBroadcaster(SBProcessRef instance);

LLDB_API const char *SBProcessGetBroadcasterClass(void);

LLDB_API bool SBProcessGetDescription(SBProcessRef instance,
                                      SBStreamRef description);

LLDB_API SBStructuredDataRef
SBProcessGetExtendedCrashInformation(SBProcessRef instance);

LLDB_API uint32_t SBProcessGetNumSupportedHardwareWatchpoints(
    SBProcessRef instance, SBErrorRef error);

LLDB_API uint32_t SBProcessLoadImage(SBProcessRef instance,
                                     SBFileSpecRef image_spec,
                                     SBErrorRef error);

LLDB_API uint32_t SBProcessLoadImageUsingPaths(SBProcessRef instance,
                                               SBFileSpecRef image_spec,
                                               SBStringListRef paths,
                                               SBFileSpecRef loaded_path,
                                               SBErrorRef error);

LLDB_API SBErrorRef SBProcessUnloadImage(SBProcessRef instance,
                                         uint32_t image_token);

LLDB_API SBErrorRef SBProcessSendEventData(SBProcessRef instance,
                                           const char *data);

LLDB_API uint32_t SBProcessGetNumExtendedBacktraceTypes(SBProcessRef instance);

LLDB_API const char *
SBProcessGetExtendedBacktraceTypeAtIndex(SBProcessRef instance, uint32_t idx);

LLDB_API SBThreadCollectionRef SBProcessGetHistoryThreads(SBProcessRef instance,
                                                          lldb_addr_t addr);

LLDB_API bool
SBProcessIsInstrumentationRuntimePresent(SBProcessRef instance,
                                         ENUM(InstrumentationRuntimeType) type);

LLDB_API SBErrorRef SBProcessSaveCore(SBProcessRef instance,
                                      const char *file_name);

LLDB_API SBErrorRef
SBProcessGetMemoryRegionInfo(SBProcessRef instance, lldb_addr_t load_addr,
                             SBMemoryRegionInfoRef region_info);

LLDB_API SBMemoryRegionInfoListRef
SBProcessGetMemoryRegions(SBProcessRef instance);

LLDB_API SBProcessInfoRef SBProcessGetProcessInfo(SBProcessRef instance);

LLDB_API lldb_addr_t SBProcessAllocateMemory(SBProcessRef instance, size_t size,
                                             uint32_t permissions,
                                             SBErrorRef error);

LLDB_API SBErrorRef SBProcessDeallocateMemory(SBProcessRef instance,
                                              lldb_addr_t ptr);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBProcessBinding_h_
