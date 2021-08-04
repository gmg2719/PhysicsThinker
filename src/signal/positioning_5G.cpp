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

#include <cstdlib>
#include <algorithm>
#include "signal/positioning_5G.h"

Sim3DCord& tdoa_positioning_4bs(int type, Sim3DCord& bs1, Sim3DCord& bs2, Sim3DCord& bs3, Sim3DCord& bs4,
                                double dt21, double dt31, double dt41)
{
    Sim3DCord *position = new Sim3DCord(0.0, 0.0, 0.0);
    double light_speed = SPEED_OF_LIGHT;
    double L = light_speed * dt21;
    double R = light_speed * dt31;
    double U = light_speed * dt41;

    if (type == TAYLOR_DIRECT_METHOD)
    {
        double XL = bs2.x_ - bs1.x_;
        double YL = bs2.y_ - bs1.y_;
        double ZL = bs2.z_ - bs1.z_;
        double XR = bs3.x_ - bs1.x_;
        double YR = bs3.y_ - bs1.y_;
        double ZR = bs3.z_ - bs1.z_;
        double XU = bs4.x_ - bs1.x_;
        double YU = bs4.y_ - bs1.y_;
        double ZU = bs4.z_ - bs1.z_;
        double E = L*L - XL*XL - YL*YL - ZL*ZL;
        double F = R*R - XR*XR - YR*YR - ZR*ZR;
        double G = U*U - XU*XU - YU*YU - ZU*ZU;
        double delta = -8 * (XL*YR*ZU+XU*YL*ZR+XR*YU*ZL-XL*YU*ZR-XR*YL*ZU-XU*YR*ZL);
        double delta1 = 4*(YR*ZU-YU*ZR);
        double delta2 = 4*(YL*ZU-YU*ZL);
        double delta3 = 4*(YL*ZR-YR*ZL);
        double MX = (2/delta)*(L*delta1-R*delta2+U*delta3);
        double NX = (1/delta)*(E*delta1-F*delta2+G*delta3);
        delta1 = 4*(XR*ZU-XU*ZR);
        delta2 = 4*(XL*ZU-XU*ZL);
        delta3 = 4*(XL*ZR-XR*ZL);
        double MY = (2/delta)*(-L*delta1+R*delta2-U*delta3);
        double NY = (1/delta)*(-E*delta1+F*delta2-G*delta3);
        delta1 = 4*(XR*YU-XU*YR);
        delta2 = 4*(XL*YU-XU*YL);
        delta3 = 4*(XL*YR-XR*YL);
        double MZ = (2/delta)*(L*delta1-R*delta2+U*delta3);
        double NZ = (1/delta)*(E*delta1-F*delta2+G*delta3);
        
        double a = MX*MX+MY*MY+MZ*MZ - 1;
        double b = 2*(MX*NX+MY*NY+MZ*NZ);
        double c = NX*NX+NY*NY+NZ*NZ;
        double k1 = (-b + sqrt(b*b-4*a*c))/(2*a);
        double k2 = (-b - sqrt(b*b-4*a*c))/(2*a);
        
        double x1 = MX*k1+NX+bs1.x_;
        double y1 = MY*k1+NY+bs1.y_;
        double z1 = MZ*k1+NZ+bs1.z_;
        double x2 = MX*k2+NX+bs1.x_;
        double y2 = MY*k2+NY+bs1.y_;
        double z2 = MZ*k2+NZ+bs1.z_;
        
        // printf("taylor_direct_solver() TDOA results (%.6f, %.6f, %.6f)\n", x1, y1, z1);
        // printf("taylor_direct_solver() TDOA results (%.6f, %.6f, %.6f)\n", x2, y2, z2);
        // printf("K = %.6f, %.6f\n", k1, k2);

        if (k2 < 0)
        {
            position->x_ = x1;
            position->y_ = y1;
            position->z_ = z1;
        } else {
            double r_ref  = sqrt((x1-bs1.x_)*(x1-bs1.x_)+(y1-bs1.y_)*(y1-bs1.y_)+(z1-bs1.z_)*(z1-bs1.z_));
            double r2_ref = sqrt((x1-bs2.x_)*(x1-bs2.x_)+(y1-bs2.y_)*(y1-bs2.y_)+(z1-bs2.z_)*(z1-bs2.z_));
            double r3_ref = sqrt((x1-bs3.x_)*(x1-bs3.x_)+(y1-bs3.y_)*(y1-bs3.y_)+(z1-bs3.z_)*(z1-bs3.z_));
            double r4_ref = sqrt((x1-bs4.x_)*(x1-bs4.x_)+(y1-bs4.y_)*(y1-bs4.y_)+(z1-bs4.z_)*(z1-bs4.z_));
            
            if (fabs(k1-k2) < 1.0)
            {
                double r_ref2  = sqrt((x2-bs1.x_)*(x2-bs1.x_)+(y2-bs1.y_)*(y2-bs1.y_)+(z2-bs1.z_)*(z2-bs1.z_));
                double r2_ref2 = sqrt((x2-bs2.x_)*(x2-bs2.x_)+(y2-bs2.y_)*(y2-bs2.y_)+(z2-bs2.z_)*(z2-bs2.z_));
                double r3_ref2 = sqrt((x2-bs3.x_)*(x2-bs3.x_)+(y2-bs3.y_)*(y2-bs3.y_)+(z2-bs3.z_)*(z2-bs3.z_));
                double r4_ref2 = sqrt((x2-bs4.x_)*(x2-bs4.x_)+(y2-bs4.y_)*(y2-bs4.y_)+(z2-bs4.z_)*(z2-bs4.z_));
                double dd1 = (r2_ref - r_ref);
                double dd2 = (r3_ref - r_ref);
                double dd3 = (r4_ref - r_ref);
                double sum1 = (dd1-L)*(dd1-L) + (dd2-R)*(dd2-R) + (dd3-U)*(dd3-U);
                double de1 = (r2_ref2 - r_ref2);
                double de2 = (r3_ref2 - r_ref2);
                double de3 = (r4_ref2 - r_ref2);
                double sum2 = (de1-L)*(de1-L) + (de2-R)*(de2-R) + (de3-U)*(de3-U);
                // printf("Least summarization = %.6e, %.6e\n", sum1, sum2);
                if ((sum1 < sum2) && (fabs((r2_ref - r_ref) - L) < 1E-4) && (fabs((r3_ref - r_ref) - R) < 1E-4) && (abs((r4_ref - r_ref) - U) < 1E-4) && (x1 >= 0) && (
                   y1 >= 0) && (z1>=0)) {
                    position->x_ = x1;
                    position->y_ = y1;
                    position->z_ = z1;
                } else {
                    position->x_ = x2;
                    position->y_ = y2;
                    position->z_ = z2;
                }
            } else {
                if ((fabs((r2_ref - r_ref) - L) < 1E-4) && (fabs((r3_ref - r_ref) - R) < 1E-4) && (abs((r4_ref - r_ref) - U) < 1E-4) && (x1 >= 0) && (
                   y1 >= 0) && (z1>=0)) {
                    position->x_ = x1;
                    position->y_ = y1;
                    position->z_ = z1;
                } else {
                    position->x_ = x2;
                    position->y_ = y2;
                    position->z_ = z2;
                }
            }
        }
    }
    else if (type == NEWTON_ITER_METHOD)
    {
        double x_est = 400.;
        double y_est = 900.;
        double z_est = 180.;
        double delta_x = 0.;
        double delta_y = 0.;
        double delta_z = 0.;

        int itr = 0;
        while (itr < NEWTON_ITER_MAXTIME)
        {
            itr += 1;

            double r1 = sqrt((x_est-bs1.x_)*(x_est-bs1.x_)+(y_est-bs1.y_)*(y_est-bs1.y_)+(z_est-bs1.z_)*(z_est-bs1.z_));
            double r2 = sqrt((x_est-bs2.x_)*(x_est-bs2.x_)+(y_est-bs2.y_)*(y_est-bs2.y_)+(z_est-bs2.z_)*(z_est-bs2.z_));
            double r3 = sqrt((x_est-bs3.x_)*(x_est-bs3.x_)+(y_est-bs3.y_)*(y_est-bs3.y_)+(z_est-bs3.z_)*(z_est-bs3.z_));
            double r4 = sqrt((x_est-bs4.x_)*(x_est-bs4.x_)+(y_est-bs4.y_)*(y_est-bs4.y_)+(z_est-bs4.z_)*(z_est-bs4.z_));
            double b1 = r2 - r1 - light_speed * dt21;
            double b2 = r3 - r1 - light_speed * dt31;
            double b3 = r4 - r1 - light_speed * dt41;
            double f11 = (1/r2) * (x_est - bs2.x_) - (1/r1)*(x_est - bs1.x_);
            double f12 = (1/r2) * (y_est - bs2.y_) - (1/r1)*(y_est - bs1.y_);
            double f13 = (1/r2) * (z_est - bs2.z_) - (1/r1)*(z_est - bs1.z_);
            double f21 = (1/r3) * (x_est - bs3.x_) - (1/r1)*(x_est - bs1.x_);
            double f22 = (1/r3) * (y_est - bs3.y_) - (1/r1)*(y_est - bs1.y_);
            double f23 = (1/r3) * (z_est - bs3.z_) - (1/r1)*(z_est - bs1.z_);
            double f31 = (1/r4) * (x_est - bs4.x_) - (1/r1)*(x_est - bs1.x_);
            double f32 = (1/r4) * (y_est - bs4.y_) - (1/r1)*(y_est - bs1.y_);
            double f33 = (1/r4) * (z_est - bs4.z_) - (1/r1)*(z_est - bs1.z_);

            double det_f = f11*f22*f33+f12*f23*f31+f13*f21*f32-f31*f22*f13-f32*f23*f11-f33*f21*f12;
            double a11 = (f22*f33-f23*f32)/det_f;
            double a12 = (f32*f13-f33*f12)/det_f;
            double a13 = (f12*f23-f13*f22)/det_f;
            double a21 = (f31*f23-f21*f33)/det_f;
            double a22 = (f11*f33-f31*f13)/det_f;
            double a23 = (f21*f13-f11*f23)/det_f;
            double a31 = (f21*f32-f31*f22)/det_f;
            double a32 = (f31*f12-f11*f32)/det_f;
            double a33 = (f11*f22-f21*f12)/det_f;
            delta_x = a11*(-b1) + a12 *(-b2) + a13*(-b3);
            delta_y = a21*(-b1) + a22 *(-b2) + a23*(-b3);
            delta_z = a31*(-b1) + a32 *(-b2) + a33*(-b3);

            if (std::max(std::max(fabs(delta_x), fabs(delta_y)), fabs(delta_z)) < 1E-6) {
                break;
            }

            x_est += delta_x;
            y_est += delta_y;
            z_est += delta_z;
            // printf("Itr %d : (%.6f %.6f %.6f) ---> (%.6f %.6f %.6f)\n", itr, delta_x, delta_y, delta_z, x_est, y_est, z_est);
        }

        position->x_ = x_est;
        position->y_ = y_est;
        position->z_ = z_est;
    }
    else
    {
        fprintf(stderr, "Not supported method for tdoa_positioning_4bs() !\n");
    }

    return (*position);
}

