//===-- SBUnixSignalsBinding.h ----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBUnixSignalsBinding_h_
#define LLDB_SBUnixSignalsBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBUnixSignalsRef CreateSBUnixSignals(void);

LLDB_API SBUnixSignalsRef CloneSBUnixSignals(SBUnixSignalsRef instance);

LLDB_API void DisposeSBUnixSignals(SBUnixSignalsRef instance);

LLDB_API void SBUnixSignalsClear(SBUnixSignalsRef instance);

LLDB_API bool SBUnixSignalsIsValid(SBUnixSignalsRef instance);

LLDB_API const char *SBUnixSignalsGetSignalAsCString(SBUnixSignalsRef instance,
                                                     int32_t signo);

LLDB_API int32_t SBUnixSignalsGetSignalNumberFromName(SBUnixSignalsRef instance,
                                                      const char *name);

LLDB_API bool SBUnixSignalsGetShouldSuppress(SBUnixSignalsRef instance,
                                             int32_t signo);

LLDB_API bool SBUnixSignalsSetShouldSuppress(SBUnixSignalsRef instance,
                                             int32_t signo, bool value);

LLDB_API bool SBUnixSignalsGetShouldStop(SBUnixSignalsRef instance,
                                         int32_t signo);

LLDB_API bool SBUnixSignalsSetShouldStop(SBUnixSignalsRef instance,
                                         int32_t signo, bool value);

LLDB_API bool SBUnixSignalsGetShouldNotify(SBUnixSignalsRef instance,
                                           int32_t signo);

LLDB_API bool SBUnixSignalsSetShouldNotify(SBUnixSignalsRef instance,
                                           int32_t signo, bool value);

LLDB_API int32_t SBUnixSignalsGetNumSignals(SBUnixSignalsRef instance);

LLDB_API int32_t SBUnixSignalsGetSignalAtIndex(SBUnixSignalsRef instance,
                                               int32_t index);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBUnixSignalsBinding_h_
