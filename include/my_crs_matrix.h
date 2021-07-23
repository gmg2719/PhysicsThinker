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

#ifndef _MY_CRS_MATRIX_H_
#define _MY_CRS_MATRIX_H_           1

// Reference to the SparseLib++
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/*             ********   ***                                 SparseLib++    */
/*          *******  **  ***       ***      ***                              */
/*           *****      ***     ******** ********                            */
/*            *****    ***     ******** ********              R. Pozo        */
/*       **  *******  ***   **   ***      ***                 K. Remington   */
/*        ********   ********                                 A. Lumsdaine   */
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
/*                                                                           */
/*                                                                           */
/*                     SparseLib++ : Sparse Matrix Library                   */
/*                                                                           */
/*               National Institute of Standards and Technology              */
/*                        University of Notre Dame                           */
/*              Authors: R. Pozo, K. Remington, A. Lumsdaine                 */
/*                                                                           */
/*                                 NOTICE                                    */
/*                                                                           */
/* Permission to use, copy, modify, and distribute this software and         */
/* its documentation for any purpose and without fee is hereby granted       */
/* provided that the above notice appear in all copies and supporting        */
/* documentation.                                                            */
/*                                                                           */
/* Neither the Institutions (National Institute of Standards and Technology, */
/* University of Notre Dame) nor the Authors make any representations about  */
/* the suitability of this software for any purpose.  This software is       */
/* provided ``as is'' without expressed or implied warranty.                 */
/*                                                                           */
/*+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

template<typename T>
class CrsMatrix
{
public:
    T *val_;        // data values (nz_ elements)
    int *rowptr_;   // row_ptr (dim_[0]+1 elements)
    int *colind_;   // col_ind (nz_ elements)

    int base_;      // index base: offset of first element
    int nz_;        // number of nonzeros
    int dim_[2];    // number of rows, cols
public:
    CrsMatrix()
    {
        val_ = NULL;
        rowptr_ = NULL;
        colind_ = NULL;
        base_ = 0;
        nz_ = 0;
        dim_[0] = 0; dim_[1] = 0;
    }

    CrsMatrix(const CrsMatrix<T> &other)
    {
        newsize(other.dim_[0], other.dim_[1], other.nz_);
        for (int i = 0; i < nz_; i++)
        {
            val_[i] = other.val_[i];
            colind_[i] = other.colind_[i];
        }
        for (int i = 0; i <= dim_[0]; i++)
        {
            rowptr_[i] = other.rowptr_[i];
        }
    }

    CrsMatrix(int m, int n, int nz, T *val, int *r, int *c, int base=0)
    {
        val_ = new T[nz];
        for (int i = 0; i < nz; i++) {
            val_[i] = val[i];
        }
        rowptr_ = new int[m+1];
        for (int i = 0; i <= m; i++) {
            rowptr_[i] = r[i];
        }
        colind_ = new int[nz];
        for (int i = 0; i < nz; i++) {
            colind_[i] = c[i];
        }
        base_ = base;
        nz_ = nz;
        dim_[0] = m; dim_[1] = n;
    }

    ~CrsMatrix()
    {
        if (val_ != NULL)     delete []val_;
        if (rowptr_ != NULL)  delete []rowptr_;
        if (colind_ != NULL)  delete []colind_;
    }

    T& val(int i) { return val_[i]; }
    int& row_ptr(int i) { return rowptr_[i]; }
    int& col_ind(int i) { return colind_[i]; }
    const T& val(int i) const { return val_[i]; }
    const int& row_ptr(int i) const { return rowptr_[i]; }
    const int& col_ind(int i) const { return colind_[i]; }

    int dim(int i) const { return dim_[i]; }
    int size(int i) const { return dim_[i]; }
    int nzeros() const { return nz_; }
    int base() const { return base_; }

    CrsMatrix<T>& operator=(const CrsMatrix<T> &other)
    {
        // Shallow copy
        val_ = other.val_;
        rowptr_ = other.rowptr_;
        colind_ = other.colind_;
        base_ = other.base_;
        nz_ = other.nz_;
        dim_[0] = other.dim_[0]; dim_[1] = other.dim_[1];
        return *this;
    }

    CrsMatrix<T>& newsize(int m, int n, int nz)
    {
        if ( dim_[0] != m) {
            if (rowptr_ != NULL)  delete []rowptr_;
            rowptr_ = new int[m+1];
        }
        dim_[0] = m;
        dim_[1] = n;
        if (nz_ != nz) {
            nz_ = nz;
            if (val_ != NULL)  delete []val_;
            val_ = new T[nz];
            if (colind_ != NULL)  delete []colind_;
            colind_ = new int[nz];
        }
        return *this;
    }

    T operator() (int i, int j) const
    {
        for (int t = rowptr_[i]; t < rowptr_[i+1]; t++)
        {
            if (colind_[t] == j) {
                return val_[t];
            }
        }

        if (i < dim_[0] && j < dim_[1]) {
            return T(0);
        } else {
            std::cerr << "Array accessing exception -- out of bounds." << "\n";
            return T(0);
        }
    }

    void print() const
    {
        
    }
};

#endif
