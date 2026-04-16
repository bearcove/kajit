//===-- SBBindingDefines.h --- ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBBindingDefines_h_
#define LLDB_SBBindingDefines_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

#ifdef BINDGEN
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(_MSC_VER)
#if defined(EXPORT_LIBLLDB)
#define LLDB_API __declspec(dllexport)
#elif defined(IMPORT_LIBLLDB)
#define LLDB_API __declspec(dllimport)
#else
#define LLDB_API
#endif
#else // defined (_MSC_VER)
#define LLDB_API
#endif

typedef uint64_t lldb_addr_t;
typedef uint64_t lldb_user_id_t;
typedef uint64_t lldb_pid_t;
typedef uint64_t lldb_tid_t;
typedef uint64_t lldb_offset_t;
typedef int32_t lldb_break_id_t;
typedef int32_t lldb_watch_id_t;
typedef uint64_t lldb_queue_id_t;

#else

#include "lldb/lldb-defines.h"
#include "lldb/lldb-enumerations.h"
#include "lldb/lldb-forward.h"
#include "lldb/lldb-types.h"

typedef lldb::addr_t lldb_addr_t;
typedef lldb::user_id_t lldb_user_id_t;
typedef lldb::pid_t lldb_pid_t;
typedef lldb::tid_t lldb_tid_t;
typedef lldb::offset_t lldb_offset_t;
typedef lldb::break_id_t lldb_break_id_t;
typedef lldb::watch_id_t lldb_watch_id_t;
typedef lldb::queue_id_t lldb_queue_id_t;

#endif

LLDB_API typedef struct SBAddressOpaque *SBAddressRef;
LLDB_API typedef struct SBAttachInfoOpaque *SBAttachInfoRef;
LLDB_API typedef struct SBBlockOpaque *SBBlockRef;
LLDB_API typedef struct SBBreakpointOpaque *SBBreakpointRef;
LLDB_API typedef struct SBBreakpointListOpaque *SBBreakpointListRef;
LLDB_API typedef struct SBBreakpointLocationOpaque *SBBreakpointLocationRef;
LLDB_API typedef struct SBBreakpointNameOpaque *SBBreakpointNameRef;
LLDB_API typedef struct SBBroadcasterOpaque *SBBroadcasterRef;
LLDB_API typedef struct SBCommandOpaque *SBCommandRef;
LLDB_API typedef struct SBCommandInterpreterOpaque *SBCommandInterpreterRef;
LLDB_API typedef struct SBCommandInterpreterRunOptionsOpaque
    *SBCommandInterpreterRunOptionsRef;
LLDB_API typedef struct SBCommandPluginInterfaceOpaque
    *SBCommandPluginInterfaceRef;
LLDB_API typedef struct SBCommandReturnObjectOpaque *SBCommandReturnObjectRef;
LLDB_API typedef struct SBCommunicationOpaque *SBCommunicationRef;
LLDB_API typedef struct SBCompileUnitOpaque *SBCompileUnitRef;
LLDB_API typedef struct SBDataOpaque *SBDataRef;
LLDB_API typedef struct SBDebuggerOpaque *SBDebuggerRef;
LLDB_API typedef struct SBDeclarationOpaque *SBDeclarationRef;
LLDB_API typedef struct SBEnvironmentOpaque *SBEnvironmentRef;
LLDB_API typedef struct SBErrorOpaque *SBErrorRef;
LLDB_API typedef struct SBEventOpaque *SBEventRef;
LLDB_API typedef struct SBEventListOpaque *SBEventListRef;
LLDB_API typedef struct SBExecutionContextOpaque *SBExecutionContextRef;
LLDB_API typedef struct SBExpressionOptionsOpaque *SBExpressionOptionsRef;
LLDB_API typedef struct SBFileOpaque *SBFileRef;
LLDB_API typedef struct SBFileSpecOpaque *SBFileSpecRef;
LLDB_API typedef struct SBFileSpecListOpaque *SBFileSpecListRef;
LLDB_API typedef struct SBFrameOpaque *SBFrameRef;
LLDB_API typedef struct SBFunctionOpaque *SBFunctionRef;
LLDB_API typedef struct SBHostOSOpaque *SBHostOSRef;
LLDB_API typedef struct SBInstructionOpaque *SBInstructionRef;
LLDB_API typedef struct SBInstructionListOpaque *SBInstructionListRef;
LLDB_API typedef struct SBLaunchInfoOpaque *SBLaunchInfoRef;
LLDB_API typedef struct SBLineEntryOpaque *SBLineEntryRef;
LLDB_API typedef struct SBListenerOpaque *SBListenerRef;
LLDB_API typedef struct SBMemoryRegionInfoOpaque *SBMemoryRegionInfoRef;
LLDB_API typedef struct SBMemoryRegionInfoListOpaque *SBMemoryRegionInfoListRef;
LLDB_API typedef struct SBModuleOpaque *SBModuleRef;
LLDB_API typedef struct SBModuleSpecOpaque *SBModuleSpecRef;
LLDB_API typedef struct SBModuleSpecListOpaque *SBModuleSpecListRef;
LLDB_API typedef struct SBPlatformOpaque *SBPlatformRef;
LLDB_API typedef struct SBProcessOpaque *SBProcessRef;
LLDB_API typedef struct SBProcessInfoOpaque *SBProcessInfoRef;
LLDB_API typedef struct SBQueueOpaque *SBQueueRef;
LLDB_API typedef struct SBQueueItemOpaque *SBQueueItemRef;
LLDB_API typedef struct SBSectionOpaque *SBSectionRef;
LLDB_API typedef struct SBSourceManagerOpaque *SBSourceManagerRef;
LLDB_API typedef struct SBStreamOpaque *SBStreamRef;
LLDB_API typedef struct SBStringListOpaque *SBStringListRef;
LLDB_API typedef struct SBStructuredDataOpaque *SBStructuredDataRef;
LLDB_API typedef struct SBSymbolOpaque *SBSymbolRef;
LLDB_API typedef struct SBSymbolContextOpaque *SBSymbolContextRef;
LLDB_API typedef struct SBSymbolContextListOpaque *SBSymbolContextListRef;
LLDB_API typedef struct SBTargetRefOpaque *SBTargetRef;
LLDB_API typedef struct SBThreadRefOpaque *SBThreadRef;
LLDB_API typedef struct SBThreadCollectionOpaque *SBThreadCollectionRef;
LLDB_API typedef struct SBThreadPlanOpaque *SBThreadPlanRef;
LLDB_API typedef struct SBTypeOpaque *SBTypeRef;
LLDB_API typedef struct SBTypeMemberOpaque *SBTypeMemberRef;
LLDB_API typedef struct SBTypeCategoryOpaque *SBTypeCategoryRef;
LLDB_API typedef struct SBTypeEnumMemberOpaque *SBTypeEnumMemberRef;
LLDB_API typedef struct SBTypeEnumMemberListOpaque *SBTypeEnumMemberListRef;
LLDB_API typedef struct SBTypeFilterOpaque *SBTypeFilterRef;
LLDB_API typedef struct SBTypeFormatOpaque *SBTypeFormatRef;
LLDB_API typedef struct SBTypeMemberFunctionOpaque *SBTypeMemberFunctionRef;
LLDB_API typedef struct SBTypeNameSpecifierOpaque *SBTypeNameSpecifierRef;
LLDB_API typedef struct SBTypeSummaryOpaque *SBTypeSummaryRef;
LLDB_API typedef struct SBTypeSummaryOptionsOpaque *SBTypeSummaryOptionsRef;
LLDB_API typedef struct SBInputReaderOpaque *SBInputReaderRef;
LLDB_API typedef struct SBPlatformConnectOptionsOpaque
    *SBPlatformConnectOptionsRef;
LLDB_API typedef struct SBPlatformShellCommandOpaque *SBPlatformShellCommandRef;

#ifndef LLDB_DISABLE_PYTHON
LLDB_API typedef struct SBTypeSyntheticOpaque *SBTypeSyntheticRef;
#endif
LLDB_API typedef struct SBTypeListOpaque *SBTypeListRef;
LLDB_API typedef struct SBValueOpaque *SBValueRef;
LLDB_API typedef struct SBValueListOpaque *SBValueListRef;
LLDB_API typedef struct SBVariablesOptionsOpaque *SBVariablesOptionsRef;
LLDB_API typedef struct SBWatchpointOpaque *SBWatchpointRef;
LLDB_API typedef struct SBUnixSignalsOpaque *SBUnixSignalsRef;

#ifdef BINDGEN
#define ENUM(name) enum name
#include "lldb/Bindings/SBBindingEnumerations.h"
#else
#define ENUM(name) enum lldb::name
#include "lldb/lldb-enumerations.h"
#endif // !BINDGEN

#endif // LLDB_SBBindingDefines_h_
