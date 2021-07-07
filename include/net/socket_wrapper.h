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

#ifndef _SOCKET_WRAPPER_H_
#define _SOCKET_WRAPPER_H_          1

//
// Implement based on the : https://github.com/BareRose/swrap
// Keep some basic interfaces and modify some contents
//

#define MY_WRAPPER_TCP      0
#define MY_WRAPPER_UDP      1

#define MY_WRAPPER_BIND     0
#define MY_WRAPPER_CONNECT  1

#define MY_WRAPPER_DEFAULT  0x00
#define MY_WRAPPER_NOBLOCK  0x01
#define MY_WRAPPER_NODELAY  0x02

typedef struct my_net_addr {
    char data[128];
} my_net_addr_t;


// Creates a new socket configured according to the given parameters
// proto ---> Allow for TCP or UDP respectively
//           MY_WRAPPER_TCP: TCP protocol connection-oriented reliable delivery
//           MY_WRAPPER_UDP: UDP protocol connectionless unreliable
// mode ---> MY_WRAPPER_BIND: Bind to given address (or all interfaces if NULL) and port, e.g. for a server
//           MY_WRAPPER_CONNECT: Connect to given address (localhost if NULL) and port, e.g. for a client
// flags ---> MY_WRAPPER_NOBLOCK: Sets the socket to be non-blocking, default is blocking
//            MY_WRAPPER_NODELAY: Disables Nagle's for TCP sockets, default is enabled
// host, serv ---> host/address as a string, can be IPv4, IPv6, etc...
//                 service/port as a string, e.g. "1728" or "http"
int my_socket_create(int proto, int mode, char flags, const char* host, const char*serv);

// Close the given socket
void my_socket_close(int sock);

// Listen for new connections with given backlog, must be MY_WRAPPER_TCP + MY_WRAPPER_BIND
int my_socket_listen(int sock, int blog);

// Use the given socket (must be my_socket_listen()) to accept a new incoming connection
int my_socket_accept(int sock, my_net_addr_t* addr);

// Writes the address with the given socket when automatically assigning port
int my_socket_ddress(int sock, my_net_addr_t*addr);

// Writes the host/address and service/port of given address into given buffers
int my_socket_address_info(my_net_addr_t* addr, char* host, int host_size, char* serv, int serv_size);

// Send given data (MY_WRAPPER_CONNECT or use my_socket_accept())
int my_socket_send(int sock, const char* data, int data_size);

// Receive data using given socket into given buffer (MY_WRAPPER_CONNECT or use my_socket_accept())
int my_socket_recv(int sock, char* data, int data_size);

// Send given data using given socket, sometimes is UDP
int my_socket_sendto(int sock, my_net_addr_t* addr, const char* data, int data_size);

// Receive data using given socket into given buffer
int my_socket_recvfrom(int sock, my_net_addr_t* addr, char* data, int data_size);

/////////////////////////////////////////////////////////////////////////////////////////////
//  Returns 1 or more if new data is available, 0 if timeout was reached, and -1 on error  //
/////////////////////////////////////////////////////////////////////////////////////////////
// Wait either until given socket has new data to receive or given time (seconds) has passed
int my_socket_select(int sock, double timeout);
// Wait either until a socket in given list has new data to receive or given time (seconds) has passed
int my_socket_multi_select(int* socks, int socks_size, double timeout);

#endif
