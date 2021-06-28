// MIT License
// 
// Copyright (c) 2021 PingzhouMing
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#ifndef _MD5_H_
#define _MD5_H_     1

#include <string>
#include <cstdint>

#if defined(__BYTE_ORDER) && (__BYTE_ORDER != 0) && (__BYTE_ORDER == __BIG_ENDIAN)
inline uint32_t md5_swap(uint32_t x)
{
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_bswap32(x);
#endif

    return (x >> 24) |
          ((x >>  8) & 0x0000FF00) |
          ((x <<  8) & 0x00FF0000) |
           (x << 24);
}
#endif

class MD5
{
public:
    // split into 64 byte blocks (=> 512 bits), hash is 16 bytes long
    enum { BLOCKSIZE = 512 / 8, HASHBYTES = 16 };

    // same as reset()
    MD5();

    // compute MD5 of a memory block
    std::string operator()(const void* data, size_t numBytes);

    // compute MD5 of a string, excluding final zero
    std::string operator()(const std::string& text);

    // add arbitrary number of bytes
    void add(const void* data, size_t numBytes);

    // return latest hash as 32 hex characters
    std::string get_hash();

    // return latest hash as bytes
    void get_hash(unsigned char buffer[HASHBYTES]);

    // restart
    void reset();

private:
    // process 64 bytes
    void process_block(const void* data);

    // process everything left in the internal buffer
    void process_buffer();

    // size of processed data in bytes
    uint64_t m_numBytes;

    // valid bytes in m_buffer
    size_t m_bufferSize;

    // bytes not processed yet
    uint8_t m_buffer[BLOCKSIZE];

    enum { HASHVALUES = HASHBYTES / 4 };

    // hash, stored as integers
    uint32_t m_hash[HASHVALUES];
};

#endif
