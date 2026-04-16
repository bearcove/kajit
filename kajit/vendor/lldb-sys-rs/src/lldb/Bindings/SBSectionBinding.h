//===-- SBSectionBinding.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBSectionBinding_h_
#define LLDB_SBSectionBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBSectionRef CreateSBSection(void);

LLDB_API SBSectionRef CloneSBSection(SBSectionRef instance);

LLDB_API void DisposeSBSection(SBSectionRef instance);

LLDB_API bool SBSectionIsValid(SBSectionRef instance);

LLDB_API const char *SBSectionGetName(SBSectionRef instance);

LLDB_API SBSectionRef SBSectionGetParent(SBSectionRef instance);

LLDB_API SBSectionRef SBSectionFindSubSection(SBSectionRef instance,
                                              const char *sect_name);

LLDB_API size_t SBSectionGetNumSubSections(SBSectionRef instance);

LLDB_API SBSectionRef SBSectionGetSubSectionAtIndex(SBSectionRef instance,
                                                    size_t idx);

LLDB_API lldb_addr_t SBSectionGetFileAddress(SBSectionRef instance);

LLDB_API lldb_addr_t SBSectionGetLoadAddress(SBSectionRef instance,
                                             SBTargetRef target);

LLDB_API lldb_addr_t SBSectionGetByteSize(SBSectionRef instance);

LLDB_API uint64_t SBSectionGetFileOffset(SBSectionRef instance);

LLDB_API uint64_t SBSectionGetFileByteSize(SBSectionRef instance);

LLDB_API SBDataRef SBSectionGetSectionData(SBSectionRef instance);

LLDB_API SBDataRef SBSectionGetSectionData2(SBSectionRef instance,
                                            uint64_t offset, uint64_t size);

LLDB_API ENUM(SectionType) SBSectionGetSectionType(SBSectionRef instance);

LLDB_API uint32_t SBSectionGetPermissions(SBSectionRef instance);

LLDB_API uint32_t SBSectionGetTargetByteSize(SBSectionRef instance);

LLDB_API uint32_t SBSectionGetAlignment(SBSectionRef instance);

LLDB_API bool SBSectionGetDescription(SBSectionRef instance,
                                      SBStreamRef description);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBSectionBinding_h_
