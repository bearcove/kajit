//===-- SBDataBinding.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SBDataBinding_h_
#define LLDB_SBDataBinding_h_

#include "lldb/Bindings/SBBindingDefines.h"

#ifdef __cplusplus
extern "C" {
#endif

LLDB_API SBDataRef CreateSBData(void);

LLDB_API SBDataRef CloneSBData(SBDataRef instance);

LLDB_API void DisposeSBData(SBDataRef instance);

LLDB_API uint8_t SBDataGetAddressByteSize(SBDataRef instance);

LLDB_API void SBDataSetAddressByteSize(SBDataRef instance,
                                       uint8_t addr_byte_size);

LLDB_API void SBDataClear(SBDataRef instance);

LLDB_API bool SBDataIsValid(SBDataRef instance);

LLDB_API size_t SBDataGetByteSize(SBDataRef instance);

LLDB_API ENUM(ByteOrder) SBDataGetByteOrder(SBDataRef instance);

LLDB_API void SBDataSetByteOrder(SBDataRef instance, ENUM(ByteOrder) endian);

LLDB_API float SBDataGetFloat(SBDataRef instance, SBErrorRef error,
                              lldb_offset_t offset);

LLDB_API double SBDataGetDouble(SBDataRef instance, SBErrorRef error,
                                lldb_offset_t offset);

LLDB_API long double SBDataGetLongDouble(SBDataRef instance, SBErrorRef error,
                                         lldb_offset_t offset);

LLDB_API lldb_addr_t SBDataGetAddress(SBDataRef instance, SBErrorRef error,
                                      lldb_offset_t offset);

LLDB_API uint8_t SBDataGetUnsignedInt8(SBDataRef instance, SBErrorRef error,
                                       lldb_offset_t offset);

LLDB_API uint16_t SBDataGetUnsignedInt16(SBDataRef instance, SBErrorRef error,
                                         lldb_offset_t offset);

LLDB_API uint32_t SBDataGetUnsignedInt32(SBDataRef instance, SBErrorRef error,
                                         lldb_offset_t offset);

LLDB_API uint64_t SBDataGetUnsignedInt64(SBDataRef instance, SBErrorRef error,
                                         lldb_offset_t offset);

LLDB_API int8_t SBDataGetSignedInt8(SBDataRef instance, SBErrorRef error,
                                    lldb_offset_t offset);

LLDB_API int16_t SBDataGetSignedInt16(SBDataRef instance, SBErrorRef error,
                                      lldb_offset_t offset);

LLDB_API int32_t SBDataGetSignedInt32(SBDataRef instance, SBErrorRef error,
                                      lldb_offset_t offset);

LLDB_API int64_t SBDataGetSignedInt64(SBDataRef instance, SBErrorRef error,
                                      lldb_offset_t offset);

LLDB_API const char *SBDataGetString(SBDataRef instance, SBErrorRef error,
                                     lldb_offset_t offset);

LLDB_API size_t SBDataReadRawData(SBDataRef instance, SBErrorRef error,
                                  lldb_offset_t offset, void *buf, size_t size);

LLDB_API bool SBDataGetDescription(SBDataRef instance, SBStreamRef description,
                                   lldb_addr_t base_addr);

LLDB_API void SBDataSetData(SBDataRef instance, SBErrorRef error, void *buf,
                            size_t size, ENUM(ByteOrder) endian,
                            uint8_t addr_size);

LLDB_API void SBDataSetDataWithOwnership(SBDataRef instance, SBErrorRef error,
                                         void *buf, size_t size,
                                         ENUM(ByteOrder) endian,
                                         uint8_t addr_size);

LLDB_API bool SBDataAppend(SBDataRef instance, SBDataRef rhs);

LLDB_API SBDataRef SBDataCreateDataFromCString(ENUM(ByteOrder) endian,
                                               uint32_t addr_byte_size,
                                               const char *data);

LLDB_API SBDataRef SBDataCreateDataFromUInt64Array(ENUM(ByteOrder) endian,
                                                   uint32_t addr_byte_size,
                                                   uint64_t *array,
                                                   size_t array_len);

LLDB_API SBDataRef SBDataCreateDataFromUInt32Array(ENUM(ByteOrder) endian,
                                                   uint32_t addr_byte_size,
                                                   uint32_t *array,
                                                   size_t array_len);

LLDB_API SBDataRef SBDataCreateDataFromSInt64Array(ENUM(ByteOrder) endian,
                                                   uint32_t addr_byte_size,
                                                   int64_t *array,
                                                   size_t array_len);

LLDB_API SBDataRef SBDataCreateDataFromSInt32Array(ENUM(ByteOrder) endian,
                                                   uint32_t addr_byte_size,
                                                   int32_t *array,
                                                   size_t array_len);

LLDB_API SBDataRef SBDataCreateDataFromDoubleArray(ENUM(ByteOrder) endian,
                                                   uint32_t addr_byte_size,
                                                   double *array,
                                                   size_t array_len);

LLDB_API bool SBDataSetDataFromCString(SBDataRef instance, const char *data);

LLDB_API bool SBDataSetDataFromUInt64Array(SBDataRef instance, uint64_t *array,
                                           size_t array_len);

LLDB_API bool SBDataSetDataFromUInt32Array(SBDataRef instance, uint32_t *array,
                                           size_t array_len);

LLDB_API bool SBDataSetDataFromSInt64Array(SBDataRef instance, int64_t *array,
                                           size_t array_len);

LLDB_API bool SBDataSetDataFromSInt32Array(SBDataRef instance, int32_t *array,
                                           size_t array_len);

LLDB_API bool SBDataSetDataFromDoubleArray(SBDataRef instance, double *array,
                                           size_t array_len);

#ifdef __cplusplus
}
#endif

#endif // LLDB_SBDataBinding_h_
