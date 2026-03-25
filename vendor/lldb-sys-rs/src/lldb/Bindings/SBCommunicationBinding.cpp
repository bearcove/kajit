//===-- SBCommunicationBinding.cpp ------------------------------*- C++ -*-===//
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

SBCommunicationRef CreateSBCommunication() {
  return reinterpret_cast<SBCommunicationRef>(new SBCommunication());
}

SBCommunicationRef CreateSBCommunication2(const char *broadcaster_name) {
  return reinterpret_cast<SBCommunicationRef>(
      new SBCommunication(broadcaster_name));
}

void DisposeSBCommunication(SBCommunicationRef instance) {
  delete reinterpret_cast<SBCommunication *>(instance);
}

bool SBCommunicationIsValid(SBCommunicationRef instance) {
  SBCommunication *unwrapped = reinterpret_cast<SBCommunication *>(instance);
  return unwrapped->IsValid();
}

SBBroadcasterRef SBCommunicationGetBroadcaster(SBCommunicationRef instance) {
  SBCommunication *unwrapped = reinterpret_cast<SBCommunication *>(instance);
  return reinterpret_cast<SBBroadcasterRef>(
      new SBBroadcaster(unwrapped->GetBroadcaster()));
}

const char *SBCommunicationGetBroadcasterClass() {
  return lldb::SBCommunication::GetBroadcasterClass();
}

enum lldb::ConnectionStatus
SBCommunicationAdoptFileDescriptor(SBCommunicationRef instance, int fd,
                                   bool owns_fd) {
  SBCommunication *unwrapped = reinterpret_cast<SBCommunication *>(instance);
  return unwrapped->AdoptFileDesriptor(fd, owns_fd);
}

enum lldb::ConnectionStatus SBCommunicationConnect(SBCommunicationRef instance,
                                                   const char *url) {
  SBCommunication *unwrapped = reinterpret_cast<SBCommunication *>(instance);
  return unwrapped->Connect(url);
}

enum lldb::ConnectionStatus
SBCommunicationDisconnect(SBCommunicationRef instance) {
  SBCommunication *unwrapped = reinterpret_cast<SBCommunication *>(instance);
  return unwrapped->Disconnect();
}

bool SBCommunicationIsConnected(SBCommunicationRef instance) {
  SBCommunication *unwrapped = reinterpret_cast<SBCommunication *>(instance);
  return unwrapped->IsConnected();
}

bool SBCommunicationGetCloseOnEOF(SBCommunicationRef instance) {
  SBCommunication *unwrapped = reinterpret_cast<SBCommunication *>(instance);
  return unwrapped->GetCloseOnEOF();
}

void SBCommunicationSetCloseOnEOF(SBCommunicationRef instance, bool b) {
  SBCommunication *unwrapped = reinterpret_cast<SBCommunication *>(instance);
  unwrapped->SetCloseOnEOF(b);
}

size_t SBCommunicationRead(SBCommunicationRef instance, void *dst,
                           size_t dst_len, uint32_t timeout_usec,
                           lldb::ConnectionStatus *status) {
  SBCommunication *unwrapped = reinterpret_cast<SBCommunication *>(instance);
  return unwrapped->Read(dst, dst_len, timeout_usec, *status);
}

size_t SBCommunicationWrite(SBCommunicationRef instance, void *src,
                            size_t src_len, lldb::ConnectionStatus *status) {
  SBCommunication *unwrapped = reinterpret_cast<SBCommunication *>(instance);
  return unwrapped->Write(src, src_len, *status);
}

bool SBCommunicationReadThreadStart(SBCommunicationRef instance) {
  SBCommunication *unwrapped = reinterpret_cast<SBCommunication *>(instance);
  return unwrapped->ReadThreadStart();
}

bool SBCommunicationReadThreadStop(SBCommunicationRef instance) {
  SBCommunication *unwrapped = reinterpret_cast<SBCommunication *>(instance);
  return unwrapped->ReadThreadStop();
}

bool SBCommunicationReadThreadIsRunning(SBCommunicationRef instance) {
  SBCommunication *unwrapped = reinterpret_cast<SBCommunication *>(instance);
  return unwrapped->ReadThreadIsRunning();
}

bool SBCommunicationSetReadThreadBytesReceivedCallback(
    SBCommunicationRef instance, ReadThreadBytesReceived callback,
    void *callback_baton) {
  SBCommunication *unwrapped = reinterpret_cast<SBCommunication *>(instance);
  return unwrapped->SetReadThreadBytesReceivedCallback(callback,
                                                       callback_baton);
}

#ifdef __cplusplus
}
#endif
