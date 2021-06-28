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
#ifndef _LINKED_LIST_H_
#define _LINKED_LIST_H_     1

#include <iostream>
#include <cstdint>
#include "basic_structure.h"

template <class T>
class LinkedList
{
private:
    uint32_t size;
    struct node<T> *start;

public:
    LinkedList()
    {
        size = 0;
        start = NULL;
    }

    struct node<T> *create_node(const T value)
    {
        struct node<T> *temp = new (struct node<T>);
        temp->next = NULL;
        temp->data = value;
        return temp;
    }

    void insert_begin(const T value)
    {
        struct node<T> *temp = create_node(value);
        temp->next = start;
        start = temp;
        size++;
    }

    void insert(const T value)
    {
        struct node<T> *temp = start;
        if (temp != NULL)
        {
            while (temp->next != NULL)
                temp = temp->next;
            temp->next = create_node(value);
            size++;
        }
        else
        {
            insert_begin(value);
        }
    }

    void insert_pos(const T value, int pos)
    {
        if (pos == 1)
        {
            insert_begin(value);
        }
        else if (pos == size + 1)
        {
            insert(value);
        }
        else if (pos <= size)
        {
            struct node<T> *curr = create_node(value), *temp = start;
            for (uint32_t i = 1; i < pos - 1; i++)
            {
                temp = temp->next;
            }
            curr->next = temp->next;
            temp->next = curr;
            size++;
        }
        else
        {
            std::cerr << "Insertion at position " << pos << " is not possible !\n";
        }
    }

    int search(const T value)
    {
        struct node<T> *temp = start;
        for (uint32_t pos = 1; temp != NULL; pos++)
        {
            if (temp->data == value)
            {
                return pos;
            }
            temp = temp->next;
        }
        return -1;
    }

    void remove_pos(const int pos)
    {
        if (pos > 1 && pos <= size)
        {
            struct node<T> *curr = NULL, *temp = start;
            for (uint32_t i = 1; i < pos - 1; i++)
            {
                temp = temp->next;
            }
            curr = temp->next;
            temp->next = curr->next;
            delete curr;
            size--;
        }
        else if (pos == 1 && size >= pos)
        {
            struct node<T> *temp = start->next;
            delete start;
            start = temp;
            size--;
        }
        else
        {
            std::cerr << "No such entry in list !\n";
        }
    }

    void remove(const T value)
    {
        remove_pos(search(value));
    }

    void reverse()
    {
        if (start != NULL && start->next != NULL)
        {
            struct node<T> *p1 = NULL, *curr = start, *p2 = start->next;
            while (p2 != NULL)
            {
                curr->next = p1;
                p1 = curr;
                curr = p2;
                p2 = p2->next;
            }
            curr->next=p1;
            start = curr;
        }
    }

    void sort()
    {
        T x;
        struct node<T>* p1=NULL, *p2=NULL, *last=NULL;

        if (start!=NULL && start->next!=NULL)
        {
            for (p1=start; p1!=NULL; p1=p1->next)
            {
                for (p2=start; p2->next!=last; p2=p2->next)
                {
                    if (p2->data > p1->data)
                    {
                        x=p1->data;
                        p1->data=p2->data;
                        p2->data=x;
                    }
                    if (p2->next==last)
                    {   last=p2;
                    }
                }
            } // End of for (p1=start:~)
        }
    }

    int getSize()
    {
        return size;
    }

    struct node<T> *getStart()
    {
        return start;
    }
};

#endif
