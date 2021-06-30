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
#ifndef _BITREADER_H_
#define _BITREADER_H_

// 
// Based on the : https://github.com/ewiger/bitreader
// BitReader reads a file as the binary mode and outputs its data as the binary string of {0, 1}
// Be similar to the "hexdump" command
//

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>

class BitReader {
private:
    size_t length;
    std::ifstream in;

    size_t get_io_stream_length(std::ifstream &stream)
    {
        size_t currentPosition = stream.tellg();
        stream.seekg(0, std::ios::end);
        size_t result = stream.tellg();
        stream.seekg(currentPosition, std::ios::beg);

        return result;
    };

    size_t get_position_in_bytes()
    {
        // get position in bytes from position in bits
        return (long) (this->index / 8);
    }

public:
    size_t index; // position in bits

    BitReader(const char* filename)
    {
        this->index = 0;
        this->in.open(filename, std::ios::in | std::ios::binary);
        if (!this->in.is_open()) {
            std::cerr << "Failed to open : " << filename << std::endl;
            exit(EXIT_FAILURE);
        }
        this->length = (size_t) this->get_io_stream_length(this->in) * 8;
    }

    /**
     * Obtain bit value at in.index
     */
    bool get_bit()
    {
        size_t position = (size_t) (this->index / 8); // bytes
        size_t state_index = this->index % 8;
        char byte = 0;
        this->assert_stream_position(position + 1);
        this->in.seekg(position, std::ios::beg);
        this->in.read(&byte, 1);
        return (byte & (1 << state_index));
    }

    bool has_ended()
    {
        return (this->index >= this->length);
    }

    void assert_stream_position(size_t position)
    {
        if (position > get_io_stream_length(in))
        {
            std::cerr << "Input stream index is out of range." << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    void static output(const char* filename)
    {
        BitReader input(filename);

        while (!input.has_ended()) {
            bool state = input.get_bit();
            const char *state_str = (state) ? "1" : "0";
            std::cout << state_str;
            input.index++;
        }
        std::cout << std::endl;
    }
};

#endif
