//===-- SBBlockBinding.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBBlockBinding_h_
#define LLDB_SBBlockBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBBlockRef CreateSBBlock(void);

LLDB_API SBBlockRef CloneSBBlock(SBBlockRef instance);

LLDB_API void DisposeSBBlock(SBBlockRef instance);

LLDB_API bool SBBlockIsInlined(SBBlockRef instance);

LLDB_API bool SBBlockIsValid(SBBlockRef instance);

LLDB_API const char *SBBlockGetInlinedName(SBBlockRef instance);

LLDB_API SBFileSpecRef SBBlockGetInlinedCallSiteFile(SBBlockRef instance);

LLDB_API uint32_t SBBlockGetInlinedCallSiteLine(SBBlockRef instance);

LLDB_API uint32_t SBBlockGetInlinedCallSiteColumn(SBBlockRef instance);

LLDB_API SBBlockRef SBBlockGetParent(SBBlockRef instance);

LLDB_API SBBlockRef SBBlockGetSibling(SBBlockRef instance);

LLDB_API SBBlockRef SBBlockGetFirstChild(SBBlockRef instance);

LLDB_API uint32_t SBBlockGetNumRanges(SBBlockRef instance);

LLDB_API SBAddressRef SBBlockGetRangeStartAddress(SBBlockRef instance,
                                                  uint32_t idx);

LLDB_API SBAddressRef SBBlockGetRangeEndAddress(SBBlockRef instance,
                                                uint32_t idx);

LLDB_API uint32_t SBBlockGetRangeIndexForBlockAddress(SBBlockRef instance,
                                                      SBAddressRef block_addr);

LLDB_API SBValueListRef SBBlockGetVariables(SBBlockRef instance,
                                            SBFrameRef frame, bool arguments,
                                            bool locals, bool statics,
                                            ENUM(DynamicValueType) use_dynamic);

LLDB_API SBValueListRef SBBlockGetVariables2(SBBlockRef instance,
                                             SBTargetRef target, bool arguments,
                                             bool locals, bool statics);

LLDB_API SBBlockRef SBBlockGetContainingInlinedBlock(SBBlockRef instance);

LLDB_API bool SBBlockGetDescription(SBBlockRef instance,
                                    SBStreamRef description);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBBlockBinding_h_
