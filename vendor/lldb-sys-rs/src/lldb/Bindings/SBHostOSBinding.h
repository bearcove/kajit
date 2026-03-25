//===-- SBHostOSBinding.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBHostOSBinding_h_
#define LLDB_SBHostOSBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBFileSpecRef SBHostOSGetProgramFileSpec(void);

LLDB_API SBFileSpecRef SBHostOSGetLLDBPythonPath(void);

LLDB_API SBFileSpecRef SBHostOSGetLLDBPath(ENUM(PathType) path_type);

LLDB_API SBFileSpecRef SBHostOSGetUserHomeDirectory(void);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBHostOSBinding_h_
