//===-- SBUnixSignalsBinding.cpp --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/API/LLDB.h"
#include "lldb/API/SBUnixSignals.h"
#include "lldb/Bindings/LLDBBinding.h"

using namespace lldb;

#ifdef __cplusplus
extern "C" {
#endif

SBUnixSignalsRef CreateSBUnixSignals() {
  return reinterpret_cast<SBUnixSignalsRef>(new SBUnixSignals());
}

SBUnixSignalsRef CloneSBUnixSignals(SBUnixSignalsRef instance) {
  return reinterpret_cast<SBUnixSignalsRef>(
      new SBUnixSignals(*reinterpret_cast<SBUnixSignals *>(instance)));
}

void DisposeSBUnixSignals(SBUnixSignalsRef instance) {
  delete reinterpret_cast<SBUnixSignals *>(instance);
}

void SBUnixSignalsClear(SBUnixSignalsRef instance) {
  SBUnixSignals *unwrapped = reinterpret_cast<SBUnixSignals *>(instance);
  unwrapped->Clear();
}

bool SBUnixSignalsIsValid(SBUnixSignalsRef instance) {
  SBUnixSignals *unwrapped = reinterpret_cast<SBUnixSignals *>(instance);
  return unwrapped->IsValid();
}

const char *SBUnixSignalsGetSignalAsCString(SBUnixSignalsRef instance,
                                            int32_t signo) {
  SBUnixSignals *unwrapped = reinterpret_cast<SBUnixSignals *>(instance);
  return unwrapped->GetSignalAsCString(signo);
}

int32_t SBUnixSignalsGetSignalNumberFromName(SBUnixSignalsRef instance,
                                             const char *name) {
  SBUnixSignals *unwrapped = reinterpret_cast<SBUnixSignals *>(instance);
  return unwrapped->GetSignalNumberFromName(name);
}

bool SBUnixSignalsGetShouldSuppress(SBUnixSignalsRef instance, int32_t signo) {
  SBUnixSignals *unwrapped = reinterpret_cast<SBUnixSignals *>(instance);
  return unwrapped->GetShouldSuppress(signo);
}

bool SBUnixSignalsSetShouldSuppress(SBUnixSignalsRef instance, int32_t signo,
                                    bool value) {
  SBUnixSignals *unwrapped = reinterpret_cast<SBUnixSignals *>(instance);
  return unwrapped->SetShouldSuppress(signo, value);
}

bool SBUnixSignalsGetShouldStop(SBUnixSignalsRef instance, int32_t signo) {
  SBUnixSignals *unwrapped = reinterpret_cast<SBUnixSignals *>(instance);
  return unwrapped->GetShouldStop(signo);
}

bool SBUnixSignalsSetShouldStop(SBUnixSignalsRef instance, int32_t signo,
                                bool value) {
  SBUnixSignals *unwrapped = reinterpret_cast<SBUnixSignals *>(instance);
  return unwrapped->SetShouldStop(signo, value);
}

bool SBUnixSignalsGetShouldNotify(SBUnixSignalsRef instance, int32_t signo) {
  SBUnixSignals *unwrapped = reinterpret_cast<SBUnixSignals *>(instance);
  return unwrapped->GetShouldNotify(signo);
}

bool SBUnixSignalsSetShouldNotify(SBUnixSignalsRef instance, int32_t signo,
                                  bool value) {
  SBUnixSignals *unwrapped = reinterpret_cast<SBUnixSignals *>(instance);
  return unwrapped->SetShouldNotify(signo, value);
}

int32_t SBUnixSignalsGetNumSignals(SBUnixSignalsRef instance) {
  SBUnixSignals *unwrapped = reinterpret_cast<SBUnixSignals *>(instance);
  return unwrapped->GetNumSignals();
}

int32_t SBUnixSignalsGetSignalAtIndex(SBUnixSignalsRef instance,
                                      int32_t index) {
  SBUnixSignals *unwrapped = reinterpret_cast<SBUnixSignals *>(instance);
  return unwrapped->GetSignalAtIndex(index);
}

#ifdef __cplusplus
}
#endif
