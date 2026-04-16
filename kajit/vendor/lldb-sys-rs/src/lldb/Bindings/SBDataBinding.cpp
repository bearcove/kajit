//===-- SBDataBinding.cpp ---------------------------------------*- C++ -*-===//
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

SBDataRef CreateSBData() { return reinterpret_cast<SBDataRef>(new SBData()); }

SBDataRef CloneSBData(SBDataRef instance) {
  return reinterpret_cast<SBDataRef>(
      new SBData(*reinterpret_cast<SBData *>(instance)));
}

void DisposeSBData(SBDataRef instance) {
  delete reinterpret_cast<SBData *>(instance);
}

uint8_t SBDataGetAddressByteSize(SBDataRef instance) {
  SBData *unwrapped = reinterpret_cast<SBData *>(instance);
  return unwrapped->GetAddressByteSize();
}

void SBDataSetAddressByteSize(SBDataRef instance, uint8_t addr_byte_size) {
  SBData *unwrapped = reinterpret_cast<SBData *>(instance);
  unwrapped->SetAddressByteSize(addr_byte_size);
}

void SBDataClear(SBDataRef instance) {
  SBData *unwrapped = reinterpret_cast<SBData *>(instance);
  unwrapped->Clear();
}

bool SBDataIsValid(SBDataRef instance) {
  SBData *unwrapped = reinterpret_cast<SBData *>(instance);
  return unwrapped->IsValid();
}

size_t SBDataGetByteSize(SBDataRef instance) {
  SBData *unwrapped = reinterpret_cast<SBData *>(instance);
  return unwrapped->GetByteSize();
}

enum lldb::ByteOrder SBDataGetByteOrder(SBDataRef instance) {
  SBData *unwrapped = reinterpret_cast<SBData *>(instance);
  return unwrapped->GetByteOrder();
}

void SBDataSetByteOrder(SBDataRef instance, lldb::ByteOrder endian) {
  SBData *unwrapped = reinterpret_cast<SBData *>(instance);
  unwrapped->SetByteOrder(endian);
}

float SBDataGetFloat(SBDataRef instance, SBErrorRef error,
                     lldb_offset_t offset) {
  SBData *unwrapped = reinterpret_cast<SBData *>(instance);
  return unwrapped->GetFloat(*reinterpret_cast<SBError *>(error), offset);
}

double SBDataGetDouble(SBDataRef instance, SBErrorRef error,
                       lldb_offset_t offset) {
  SBData *unwrapped = reinterpret_cast<SBData *>(instance);
  return unwrapped->GetDouble(*reinterpret_cast<SBError *>(error), offset);
}

long double SBDataGetLongDouble(SBDataRef instance, SBErrorRef error,
                                lldb_offset_t offset) {
  SBData *unwrapped = reinterpret_cast<SBData *>(instance);
  return unwrapped->GetLongDouble(*reinterpret_cast<SBError *>(error), offset);
}

lldb_addr_t SBDataGetAddress(SBDataRef instance, SBErrorRef error,
                             lldb_offset_t offset) {
  SBData *unwrapped = reinterpret_cast<SBData *>(instance);
  return unwrapped->GetAddress(*reinterpret_cast<SBError *>(error), offset);
}

uint8_t SBDataGetUnsignedInt8(SBDataRef instance, SBErrorRef error,
                              lldb_offset_t offset) {
  SBData *unwrapped = reinterpret_cast<SBData *>(instance);
  return unwrapped->GetUnsignedInt8(*reinterpret_cast<SBError *>(error),
                                    offset);
}

uint16_t SBDataGetUnsignedInt16(SBDataRef instance, SBErrorRef error,
                                lldb_offset_t offset) {
  SBData *unwrapped = reinterpret_cast<SBData *>(instance);
  return unwrapped->GetUnsignedInt16(*reinterpret_cast<SBError *>(error),
                                     offset);
}

uint32_t SBDataGetUnsignedInt32(SBDataRef instance, SBErrorRef error,
                                lldb_offset_t offset) {
  SBData *unwrapped = reinterpret_cast<SBData *>(instance);
  return unwrapped->GetUnsignedInt32(*reinterpret_cast<SBError *>(error),
                                     offset);
}

uint64_t SBDataGetUnsignedInt64(SBDataRef instance, SBErrorRef error,
                                lldb_offset_t offset) {
  SBData *unwrapped = reinterpret_cast<SBData *>(instance);
  return unwrapped->GetUnsignedInt64(*reinterpret_cast<SBError *>(error),
                                     offset);
}

int8_t SBDataGetSignedInt8(SBDataRef instance, SBErrorRef error,
                           lldb_offset_t offset) {
  SBData *unwrapped = reinterpret_cast<SBData *>(instance);
  return unwrapped->GetSignedInt8(*reinterpret_cast<SBError *>(error), offset);
}

int16_t SBDataGetSignedInt16(SBDataRef instance, SBErrorRef error,
                             lldb_offset_t offset) {
  SBData *unwrapped = reinterpret_cast<SBData *>(instance);
  return unwrapped->GetSignedInt16(*reinterpret_cast<SBError *>(error), offset);
}

int32_t SBDataGetSignedInt32(SBDataRef instance, SBErrorRef error,
                             lldb_offset_t offset) {
  SBData *unwrapped = reinterpret_cast<SBData *>(instance);
  return unwrapped->GetSignedInt32(*reinterpret_cast<SBError *>(error), offset);
}

int64_t SBDataGetSignedInt64(SBDataRef instance, SBErrorRef error,
                             lldb_offset_t offset) {
  SBData *unwrapped = reinterpret_cast<SBData *>(instance);
  return unwrapped->GetSignedInt64(*reinterpret_cast<SBError *>(error), offset);
}

const char *SBDataGetString(SBDataRef instance, SBErrorRef error,
                            lldb_offset_t offset) {
  SBData *unwrapped = reinterpret_cast<SBData *>(instance);
  return unwrapped->GetString(*reinterpret_cast<SBError *>(error), offset);
}

size_t SBDataReadRawData(SBDataRef instance, SBErrorRef error,
                         lldb_offset_t offset, void *buf, size_t size) {
  SBData *unwrapped = reinterpret_cast<SBData *>(instance);
  return unwrapped->ReadRawData(*reinterpret_cast<SBError *>(error), offset,
                                buf, size);
}

bool SBDataGetDescription(SBDataRef instance, SBStreamRef description,
                          lldb_addr_t base_addr) {
  SBData *unwrapped = reinterpret_cast<SBData *>(instance);
  return unwrapped->GetDescription(*reinterpret_cast<SBStream *>(description),
                                   base_addr);
}

void SBDataSetData(SBDataRef instance, SBErrorRef error, void *buf, size_t size,
                   lldb::ByteOrder endian, uint8_t addr_size) {
  SBData *unwrapped = reinterpret_cast<SBData *>(instance);
  unwrapped->SetData(*reinterpret_cast<SBError *>(error), buf, size, endian,
                     addr_size);
}

void SBDataSetDataWithOwnership(SBDataRef instance, SBErrorRef error, void *buf,
                                size_t size, lldb::ByteOrder endian,
                                uint8_t addr_size) {
  SBData *unwrapped = reinterpret_cast<SBData *>(instance);
  unwrapped->SetDataWithOwnership(*reinterpret_cast<SBError *>(error), buf,
                                  size, endian, addr_size);
}

bool SBDataAppend(SBDataRef instance, SBDataRef rhs) {
  SBData *unwrapped = reinterpret_cast<SBData *>(instance);
  return unwrapped->Append(*reinterpret_cast<SBData *>(rhs));
}

SBDataRef SBDataCreateDataFromCString(lldb::ByteOrder endian,
                                      uint32_t addr_byte_size,
                                      const char *data) {
  return reinterpret_cast<SBDataRef>(new SBData(
      lldb::SBData::CreateDataFromCString(endian, addr_byte_size, data)));
}

SBDataRef SBDataCreateDataFromUInt64Array(lldb::ByteOrder endian,
                                          uint32_t addr_byte_size,
                                          uint64_t *array, size_t array_len) {
  return reinterpret_cast<SBDataRef>(
      new SBData(lldb::SBData::CreateDataFromUInt64Array(endian, addr_byte_size,
                                                         array, array_len)));
}

SBDataRef SBDataCreateDataFromUInt32Array(lldb::ByteOrder endian,
                                          uint32_t addr_byte_size,
                                          uint32_t *array, size_t array_len) {
  return reinterpret_cast<SBDataRef>(
      new SBData(lldb::SBData::CreateDataFromUInt32Array(endian, addr_byte_size,
                                                         array, array_len)));
}

SBDataRef SBDataCreateDataFromSInt64Array(lldb::ByteOrder endian,
                                          uint32_t addr_byte_size,
                                          int64_t *array, size_t array_len) {
  return reinterpret_cast<SBDataRef>(
      new SBData(lldb::SBData::CreateDataFromSInt64Array(endian, addr_byte_size,
                                                         array, array_len)));
}

SBDataRef SBDataCreateDataFromSInt32Array(lldb::ByteOrder endian,
                                          uint32_t addr_byte_size,
                                          int32_t *array, size_t array_len) {
  return reinterpret_cast<SBDataRef>(
      new SBData(lldb::SBData::CreateDataFromSInt32Array(endian, addr_byte_size,
                                                         array, array_len)));
}

SBDataRef SBDataCreateDataFromDoubleArray(lldb::ByteOrder endian,
                                          uint32_t addr_byte_size,
                                          double *array, size_t array_len) {
  return reinterpret_cast<SBDataRef>(
      new SBData(lldb::SBData::CreateDataFromDoubleArray(endian, addr_byte_size,
                                                         array, array_len)));
}

bool SBDataSetDataFromCString(SBDataRef instance, const char *data) {
  SBData *unwrapped = reinterpret_cast<SBData *>(instance);
  return unwrapped->SetDataFromCString(data);
}

bool SBDataSetDataFromUInt64Array(SBDataRef instance, uint64_t *array,
                                  size_t array_len) {
  SBData *unwrapped = reinterpret_cast<SBData *>(instance);
  return unwrapped->SetDataFromUInt64Array(array, array_len);
}

bool SBDataSetDataFromUInt32Array(SBDataRef instance, uint32_t *array,
                                  size_t array_len) {
  SBData *unwrapped = reinterpret_cast<SBData *>(instance);
  return unwrapped->SetDataFromUInt32Array(array, array_len);
}

bool SBDataSetDataFromSInt64Array(SBDataRef instance, int64_t *array,
                                  size_t array_len) {
  SBData *unwrapped = reinterpret_cast<SBData *>(instance);
  return unwrapped->SetDataFromSInt64Array(array, array_len);
}

bool SBDataSetDataFromSInt32Array(SBDataRef instance, int32_t *array,
                                  size_t array_len) {
  SBData *unwrapped = reinterpret_cast<SBData *>(instance);
  return unwrapped->SetDataFromSInt32Array(array, array_len);
}

bool SBDataSetDataFromDoubleArray(SBDataRef instance, double *array,
                                  size_t array_len) {
  SBData *unwrapped = reinterpret_cast<SBData *>(instance);
  return unwrapped->SetDataFromDoubleArray(array, array_len);
}

#ifdef __cplusplus
}
#endif
