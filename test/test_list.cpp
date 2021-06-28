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
#include "common/doubly_linked_list.hpp"

int main(void)
{
    Doubly_LinkedList<int> ll;
    ll.insert_begin(5);
    ll.insert_begin(6);
    ll.insert(78);
    ll.insert_pos(54, 4);
    ll.insert_pos(8, 4);
    ll.insert_pos(7, 1);
    ll.insert_pos(8888, 12);
    //7 6 5 78 8 54
    struct node<int> *temp = ll.getStart();
    for (; temp != NULL; temp = temp->next)
        cout << temp->data << "\n";
    cout << "size=" << ll.getSize() << "\n";
    cout << "78 found at position: " << ll.search(78) << "\n";
    ll.remove(78);
    ll.remove_pos(5);
    ll.remove_pos(1);
    ll.remove_pos(10);
    //6 5 8
    temp = ll.getStart();
    for (; temp != NULL; temp = temp->next)
        cout << temp->data << "\n";
    cout << "size=" << ll.getSize() << "\n";
    ll.reverse();
    temp = ll.getStart();
    for (; temp != NULL; temp = temp->next)
        cout << temp->data << "\n";
    cout << "size=" << ll.getSize() << "\n";
    ll.insert(-1);
    ll.insert(34);
    ll.sort();
    //-1 5 6 8 34
    temp = ll.getStart();
    for (; temp != NULL; temp = temp->next)
        cout << temp->data << "\n";
    cout << "size=" << ll.getSize() << "\n";

    return 0;
}
