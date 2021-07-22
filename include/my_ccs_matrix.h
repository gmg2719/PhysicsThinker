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

#ifndef _MY_CCS_MATRIX_H_
#define _MY_CCS_MATRIX_H_           1

template<typename T>
class CcsMatrix
{
public:
    T *val_;        // data values (nz_ elements)
    int *rowind_;   // row_ind (nz_ elements)
    int *colptr_;   // col_ptr (dim_[1]+1 elements)

    int base_;      // index base: offset of first element
    int nz_;        // number of nonzeros
    int dim_[2];    // number of rows, cols
public:
    CcsMatrix();
    CcsMatrix(const CcsMatrix<T> &other);
    CcsMatrix(int m, int n, int nz, T *val, int *r, int *c, int base=0);
    ~CcsMatrix();

    T& val(int i) { return val_[i]; }
    int& row_ind(int i) { return rowind_[i]; }
    int& col_ptr(int i) { return colptr_[i]; }
    const T& val(int i) const { return val_[i]; }
    const int& row_ind(int i) const { return rowind_[i]; }
    const int& col_ptr(int i) const { return colptr_[i]; }

    int dim(int i) const { return dim_[i]; }
    int size(int i) const { return dim_[i]; }
    int nzeros() const { return nz_; }
    int base() const { return base_; }

    CrsMatrix<T>& operator=(const CrsMatrix<T> &r);
    CrsMatrix<T>& newsize(int m, int n, int nz);

    T operator() (int i, int j) const;
    T& set(int i, int j);


    void print() const;
};

#endif
