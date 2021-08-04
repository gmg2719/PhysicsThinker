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

#ifndef _MY_PHYSICS_CONSTANTS_H_
#define _MY_PHYSICS_CONSTANTS_H_            1

#if defined(__cplusplus) && (__cplusplus >= 201103L)

const double GOLDEN_RATIO       = 1.618033988749895;
const double SPEED_OF_LIGHT     = 299792458.0;
const double PLANCK_CONSTANT    = 6.62607015E-34;
const double BOLTZMANN_CONSTANT = 1.380658E-23;
const double GRAVITY_CONSTANT   = 9.80665;
const double AVOGADRO_CONSTANT  = 6.02214076E+23;
const double ELECTRON_MASS      = 9.1093837015E-31;
const double PROTON_MASS        = 1.67262192369E-27;
const double NEUTRON_MASS       = 1.67492749804E-27;
const double ATOMIC_MASS        = 1.6605390666E-27;

#else

#define GOLDEN_RATIO                (1.618033988749895d)
#define SPEED_OF_LIGHT              (299792458.0d)
#define PLANCK_CONSTANT             ((6.62607015E-34)d)
#define BOLTZMANN_CONSTANT          ((1.380658E-23)d)
#define GRAVITY_CONSTANT            (9.80665d)
#define AVOGADRO_CONSTANT           ((6.02214076E+23)d)
#define ELECTRON_MASS               ((9.1093837015E-31)d)
#define PROTON_MASS                 ((1.67262192369E-27)d)
#define NEUTRON_MASS                ((1.67492749804E-27)d)
#define ATOMIC_MASS                 ((1.6605390666E-27)d)

#endif

#endif

