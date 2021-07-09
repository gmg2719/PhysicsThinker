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

#include <sys/socket.h>
#include <netdb.h>
#include <fcntl.h>
#include <unistd.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include "net/socket_wrapper.h"

int my_socket_create(int proto, int mode, char flags, const char* host, const char*serv)
{
    struct addrinfo* result, hint = {
        (mode == MY_WRAPPER_BIND) ? AI_PASSIVE : 0, //ai_flags
        AF_UNSPEC, //ai_family
        (proto == MY_WRAPPER_TCP) ? SOCK_STREAM : SOCK_DGRAM, //ai_socktype
        0, 0, NULL, NULL, NULL};
    //get address info
    if (getaddrinfo(host, serv, &hint, &result)) return -1;
    //create socket
    int sock = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
    if (sock == -1) return -1;

    //make sure IPV6_ONLY is disabled
    if (result->ai_family == AF_INET6) {
        int no = 0;
        setsockopt(sock, IPPROTO_IPV6, IPV6_V6ONLY, (void*)&no, sizeof(no));
    }
    //set TCP_NODELAY if applicable
    if (proto == MY_WRAPPER_TCP) {
        int nodelay = (flags&MY_WRAPPER_NODELAY);
        setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, (void*)&nodelay, sizeof(nodelay));
    }
    //bind if applicable
    if ((mode == MY_WRAPPER_BIND)&&(bind(sock, result->ai_addr, result->ai_addrlen))) {
        my_socket_close(sock);
        return -1;
    }
    //set non-blocking if needed
    if (flags&MY_WRAPPER_NOBLOCK) {
        if (fcntl(sock, F_SETFL, O_NONBLOCK, 1) == -1) {
            my_socket_close(sock);
            return -1;
        }
    }
    //connect if applicable (return only relevant if blocking)
    if ((mode == MY_WRAPPER_CONNECT)&&(connect(sock, result->ai_addr, result->ai_addrlen))&&(!(flags&MY_WRAPPER_NOBLOCK))) {
        my_socket_close(sock);
        return -1;
    }
    //free address info
    freeaddrinfo(result);
    //return socket handle
    return sock;
}

void my_socket_close(int sock)
{
    close(sock);
}

int my_socket_listen(int sock, int blog)
{
    return listen(sock, blog);
}

int my_socket_accept(int sock, my_net_addr_t* addr)
{
    socklen_t addr_size = sizeof(struct my_net_addr);
    return accept(sock, (struct sockaddr*)addr, (addr) ? &addr_size : NULL);
}

int my_socket_ddress(int sock, my_net_addr_t*addr)
{
    socklen_t addr_size = sizeof(struct my_net_addr);
    return getsockname(sock, (struct sockaddr*)addr, &addr_size);
}

int my_socket_address_info(my_net_addr_t* addr, char* host, int host_size, char* serv, int serv_size)
{
    return getnameinfo((struct sockaddr*)addr, sizeof(struct my_net_addr), host, host_size, serv, serv_size, 0);
}

int my_socket_send(int sock, const char* data, int data_size)
{
    return send(sock, data, data_size, 0);
}

int my_socket_recv(int sock, char* data, int data_size)
{
    return recv(sock, data, data_size, 0);
}

int my_socket_sendto(int sock, my_net_addr_t* addr, const char* data, int data_size)
{
    return sendto(sock, data, data_size, 0, (struct sockaddr*)addr, sizeof(struct my_net_addr));
}

int my_socket_recvfrom(int sock, my_net_addr_t* addr, char* data, int data_size)
{
    socklen_t addr_size = sizeof(struct my_net_addr);
    return recvfrom(sock, data, data_size, 0, (struct sockaddr*)addr, &addr_size);
}

int my_socket_select(int sock, double timeout)
{
    fd_set set;
    struct timeval time;
    //fd set
    FD_ZERO(&set);
    if (sock > -1) FD_SET(sock, &set);
    //timeout
    time.tv_sec = timeout;
    time.tv_usec = (timeout - time.tv_sec)*1000000.0;
    //return
    return select(sock+1, &set, NULL, NULL, &time);
}

int my_socket_multi_select(int* socks, int socks_size, double timeout)
{
    fd_set set;
    struct timeval time;
    int sock_max = -1;
    //fd set
    FD_ZERO(&set);
    for (int i = 0; i < socks_size; i++) {
        if (socks[i] > sock_max) sock_max = socks[i];
        if (socks[i] > -1) FD_SET(socks[i], &set);
    }
    //timeout
    time.tv_sec = timeout;
    time.tv_usec = (timeout - time.tv_sec)*1000000.0;
    //return
    return select(sock_max+1, &set, NULL, NULL, &time);
}

