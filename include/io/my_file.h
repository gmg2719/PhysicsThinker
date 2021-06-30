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
#ifndef _MY_FILE_H_
#define _MY_FILE_H_     1

#include <cstddef>

//
// Implement and modify based on the : https://github.com/jad-darrous/FileIO
//

#define MY_MAX_BUFFER_SIZE  (1024000UL)
#define READ_MODE           1
#define WRITE_MODE          2
#define OPENNED             1
#define CLOSED              2

typedef struct my_file
{
    char *buffer[MY_MAX_BUFFER_SIZE];
    int fd;
    int buf_p;      // The current reading index, read mode
    int buf_len;    // The count of bytes into the buffer
    int _size;      // The count of read bytes till now, read mode
    int size;       // The total size (bytes) of the file
    int mode;       // READ_MODE or WRITE_MODE
    int stat;       // File open or closed state
    int eof;        // If and EOF has been encountered during one previous reading
} my_file_t;

void my_debug_printfile(my_file_t *f);
my_file_t *my_fopen(char *name, const char *mode);
int my_fclose(my_file_t *f);
int my_fread(my_file_t *f, void *p, size_t size, size_t nbelem);
int my_fwrite(my_file_t *f, void *p, size_t size, size_t nbelem);
int my_feof(my_file_t *f);
int my_file_flush(my_file_t *f);
int my_file_size(my_file_t *f);
int my_fscanf(my_file_t *f, const char *format, ...);
int my_fprintf(my_file_t *f, const char *format, ...);

#endif
