//===-- SBCommunicationBinding.h --------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBCommunicationBinding_h_
#define LLDB_SBCommunicationBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

typedef void (*ReadThreadBytesReceived)(void *baton, const void *src,
                                        size_t src_len);

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBCommunicationRef CreateSBCommunication(void);

LLDB_API SBCommunicationRef
CreateSBCommunication2(const char *broadcaster_name);

LLDB_API void DisposeSBCommunication(SBCommunicationRef instance);

LLDB_API bool SBCommunicationIsValid(SBCommunicationRef instance);

LLDB_API SBBroadcasterRef
SBCommunicationGetBroadcaster(SBCommunicationRef instance);

LLDB_API const char *SBCommunicationGetBroadcasterClass(void);

LLDB_API ENUM(ConnectionStatus)
    SBCommunicationAdoptFileDescriptor(SBCommunicationRef instance, int fd,
                                       bool owns_fd);

LLDB_API ENUM(ConnectionStatus)
    SBCommunicationConnect(SBCommunicationRef instance, const char *url);

LLDB_API ENUM(ConnectionStatus)
    SBCommunicationDisconnect(SBCommunicationRef instance);

LLDB_API bool SBCommunicationIsConnected(SBCommunicationRef instance);

LLDB_API bool SBCommunicationGetCloseOnEOF(SBCommunicationRef instance);

LLDB_API void SBCommunicationSetCloseOnEOF(SBCommunicationRef instance, bool b);

LLDB_API size_t SBCommunicationRead(SBCommunicationRef instance, void *dst,
                                    size_t dst_len, uint32_t timeout_usec,
                                    ENUM(ConnectionStatus) * status);

LLDB_API size_t SBCommunicationWrite(SBCommunicationRef instance, void *src,
                                     size_t src_len,
                                     ENUM(ConnectionStatus) * status);

LLDB_API bool SBCommunicationReadThreadStart(SBCommunicationRef instance);

LLDB_API bool SBCommunicationReadThreadStop(SBCommunicationRef instance);

LLDB_API bool SBCommunicationReadThreadIsRunning(SBCommunicationRef instance);

LLDB_API bool SBCommunicationSetReadThreadBytesReceivedCallback(
    SBCommunicationRef instance, ReadThreadBytesReceived callback,
    void *callback_baton);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBCommunicationBinding_h_
