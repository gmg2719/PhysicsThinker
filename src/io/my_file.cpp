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
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include "io/my_file.h"

// A system call to read from the file
void _read_file(my_file_t *f)
{
    f->buf_len = read(f->fd, f->buffer, MY_MAX_BUFFER_SIZE);
    f->buf_p = 0;
}

// A system call to write to the file
void _write_file(my_file_t *f)
{
    write(f->fd, (char*)f->buffer, f->buf_p);
    f->size += f->buf_p;
    f->buf_p = 0;
}

void _reput_char(my_file_t *f, char c)
{
    ((char*)f->buffer)[--f->buf_p] = c;
    f->_size--;
}

char _get_first_non_empty_char(my_file_t *f)
{
    int result;
    char chr;
    while ((result = my_fread(f, &chr, sizeof(char), 1)) == 1 && isspace(chr));
    return (result == 1 ? chr : 0);
}

// Debugging
void my_debug_printfile(my_file_t *f)
{
    printf("MY_FILE: %p  fd=%d mode=%d size=%d buf_p=%d buf_len=%d\n",
           f, f->fd, f->mode, f->size, f->buf_p, f->buf_len);
}

//
// Create a my_file_t object and initialize it:
//     reading mode: make the first read syscall
//     writing mode: create file if it doesn't exist
//
my_file_t *my_fopen(char *name, const char *mode)
{
    my_file_t *f = new my_file_t;

    if (!strncmp(mode, "r", 1)) {
        f->mode = READ_MODE;
        if ((f->fd = open(name, O_RDONLY, S_IRWXU)) == -1) {
            return NULL;
        }
        _read_file(f);
        struct stat buf;
        fstat(f->fd, &buf);
        f->size = buf.st_size;
        f->_size = 0;

    } else if (!strncmp(mode, "w", 1)) {
        f->mode = WRITE_MODE;
        if ((f->fd = open(name, O_WRONLY | O_CREAT | O_TRUNC, S_IRWXU)) == -1) {
            return NULL;
        }
        f->size = f->buf_p = 0;

    } else {
        fprintf(stderr, "openning mode err\n");
        return NULL;
    }
    f->eof = 0;
    f->stat = OPENNED;
    return f;
}

int my_fclose(my_file_t *f)
{
    if (f == NULL)  return -1;
    if (f->stat == CLOSED)
    {
        return -1;
    }
    if (f->mode == WRITE_MODE)
    {
        _write_file(f);
    }
    f->stat = CLOSED;
    int fd = f->fd;
    free(f);
    return close(fd);
}

int my_fread(my_file_t *f, void *p, size_t size, size_t nbelem)
{
    if (f->stat == CLOSED || f->mode == WRITE_MODE) {
        return -1;
    }
    if (f->size == f->_size) {
        f->eof = 1;
    }
    if (my_feof(f)) {
        return 0;
    }
    /* total bytes to read */
    int n = size * nbelem;
    /* the current read index in the user buffer */
    int p_ind = 0;
    while (f->buf_p + n > f->buf_len) {
        /* nb of bytesW that are buffered but not yet copied to the user buffer */
        int rem_buf = f->buf_len - f->buf_p;
        if (rem_buf) {
            memcpy((char*)p + p_ind, (char*) f->buffer + f->buf_p, rem_buf);
            f->buf_p += rem_buf;
            f->_size += rem_buf;
            p_ind += rem_buf;

            if (f->size == f->_size) return p_ind / size;
        }
        _read_file(f);
        if (f->buf_len == -1) {
            return -1;
        }
        if (f->buf_len == 0) {
            f->eof = 1;
            return p_ind / size;
        }
        n -= rem_buf;
    }
    memcpy((char*)p + p_ind, (char*) f->buffer + f->buf_p, n);
    f->buf_p += n;
    f->_size += n;
    return nbelem;
}

int my_fwrite(my_file_t *f, void *p, size_t size, size_t nbelem)
{
    if (f->stat == CLOSED || f->mode == READ_MODE) {
        return -1;
    }
    /* total bytes to write */
    int n = size * nbelem;
    /* the current write index in the user buffer */
    int p_ind = 0;
    while (f->buf_p + n > MY_MAX_BUFFER_SIZE) {
        int empty_buf = MY_MAX_BUFFER_SIZE - f->buf_p;
        memcpy((char*)f->buffer + f->buf_p, (char*)p + p_ind, empty_buf);
        f->buf_p += empty_buf;
        _write_file(f);
        n -= empty_buf;
        p_ind += empty_buf;
    }
    memcpy((char*)f->buffer + f->buf_p, (char*)p + p_ind, n);
    f->buf_p += n;
    return nbelem;
}

// is end of file (reading mode)
int my_feof(my_file_t *f)
{
    return f->eof;
}

// Flush the buffer content to the disk (writing mode)W
int my_file_flush(my_file_t *f)
{
    if (f->stat == CLOSED || f->mode == READ_MODE)
    {
        return -1;
    }
    int bytes = f->buf_len;
    _write_file(f);
    return bytes;
}

// Returns the total size of a file if openned in reading mode
// Or the number of flushed bytes to disk of a file if openned
// in writing mode
int my_file_size(my_file_t *f)
{
    return f->size;
}

// formated read - std scanf style
int my_fscanf(my_file_t *f, const char *format, ...)
{
    if (f->stat == CLOSED || f->mode == WRITE_MODE) {
        return -1;
    }
    if (my_feof(f)) {
		return 0;
	}
    /* the number of arguments successfully written */
    int c = 0;
    /* the list holing the args */
    va_list arguments;
    /* Initializing arguments to store all values after format */
    va_start(arguments, format);
    /* Sum all the inputs; we still rely on the function caller to tell us how
     * many there are */
    const char *str = format;
    while (*str) {
        if (*str == '%') {
            str++;
            char tt = *str, chr;
            if (tt == 'c') {
                void* ch = va_arg(arguments, void*);
                if (my_fread(f, ch, sizeof(char), 1))
                    c++;
            } else if (tt == 'd') {
                int* x = va_arg(arguments, int*);
                chr = _get_first_non_empty_char(f);
                if (chr == 0) return -1;
                char num[12] = {chr};
                int num_p = 1;
                while (my_fread(f, &chr, sizeof(char), 1) == 1 && isdigit(chr) && num_p < 12) {
                    num[num_p++] = chr;
                }
                _reput_char(f, chr);
                *x = atoi(num);
                c++;
            } else if (tt == 's') {
                char* ss = va_arg(arguments, char*);
                char* p = ss;
                *p = _get_first_non_empty_char(f);
                p++;
                while (my_fread(f, &chr, sizeof(char), 1) == 1 && !isspace(chr)) {
                    *p = chr;
                    p++;
                }
                /* append the NULL character to the string */
                *p = 0;
                _reput_char(f, chr);
                c++;
            } else {
                return -1;
            }
        } else {
        }
        str++;
    }
    va_end(arguments);
    return c;
}

// formated write - std printf style
int my_fprintf(my_file_t *f, const char *format, ...)
{
    if (f->stat == CLOSED || f->mode == READ_MODE) {
        return -1;
    }
    /* the number of arguments successfully written */
    int c = 0;
    /* the list holing the args */
    va_list arguments;
    /* Initializing arguments to store all values after format */
    va_start(arguments, format);
    /* Sum all the inputs; we still rely on the function caller to tell us how
     * many there are */
    const char *str = &format[0];
    while (*str) {
        if (*str == '%') {
            str++;
            char tt = *str;
            if (tt == 'c') {
                char ch = va_arg(arguments, int);
                if (my_fwrite(f, &ch, sizeof(char), 1))
                    c++;
            } else if (tt == 'd') {
                int x = va_arg(arguments, int);
                char bb[12] = {'\0'};
                sprintf(bb, "%d", x);
                if (my_fwrite(f, bb, strlen(bb), 1))
                    c++;
            } else if (tt == 's') {
                char* ss = va_arg(arguments, char*);
                if (my_fwrite(f, ss, strlen(ss), 1))
                    c++;
            } else {
                return -1;
            }
        } else {
            char ch = *str;
            my_fwrite(f, &ch, sizeof(char), 1);
        }
        str++;
    }
    va_end(arguments);
    return c;
}
