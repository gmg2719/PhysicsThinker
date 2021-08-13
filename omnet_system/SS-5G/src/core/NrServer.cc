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

#include "NrServer.h"
#include "../message/corepacket_m.h"

using namespace omnetpp;

namespace ss5G {

Define_Module(NrServer);

static int server_id_omnetpp = 10000;

void NrServer::initialize()
{
    position_do_event = nullptr;
    cMessage *msg = new cMessage("positiong-do");
    position_do_event = msg;

    scheduleAt(10.0, position_do_event);
    EV << "Scheduling position doing on the Server\n";
}

void NrServer::handleMessage(cMessage *msg)
{
    if (msg->isSelfMessage()) {
        if (msg == position_do_event) {
            EV << "Perform positioning operation for the UE list table !\n";
            tdoa_positioning();

            // Every 100 ms, there will perform the positioning in the server !
            scheduleAt(simTime() + 0.1, position_do_event);
        } else {
            delete msg;
        }
    } else {
        CorePacket *coremsg = check_and_cast<CorePacket *>(msg);

        if (coremsg->getType() == BS_POSITION_REQ)
        {
            receive_positioning_request(coremsg);
        }

        delete msg;
    }
}

void NrServer::receive_positioning_request(CorePacket *coremsg)
{
    int bs_simid = coremsg->getBsSimId();
    int ue_simid = coremsg->getUeSimId();
    std::map<int, ue_position_info_t>::iterator it;

    it = ue_position_table.find(ue_simid);

    if (it == ue_position_table.end()) {
        // Not found
        ue_position_info_t e;
        ue_coordinate_t received_pos;
        int port = coremsg->getArrivalGate()->getIndex();

        e.sim_id = ue_simid;
        e.bs_from[port].x = coremsg->getBsX();
        e.bs_from[port].y = coremsg->getBsY();
        e.bs_from[port].z = coremsg->getBsZ();
        e.arrive_time[port] = coremsg->getArriveTime();

        received_pos.x = coremsg->getUeX();
        received_pos.y = coremsg->getUeY();
        received_pos.z = coremsg->getUeZ();
        e.coord.push_back(received_pos);

        ue_position_table.insert(std::pair<int, ue_position_info_t>(ue_simid, e));
    } else {
        ue_coordinate_t received_pos;
        int port = coremsg->getArrivalGate()->getIndex();

        it->second.bs_from[port].x = coremsg->getBsX();
        it->second.bs_from[port].y = coremsg->getBsY();
        it->second.bs_from[port].z = coremsg->getBsZ();
        it->second.arrive_time[port] = coremsg->getArriveTime();

        received_pos.x = coremsg->getUeX();
        received_pos.y = coremsg->getUeY();
        received_pos.z = coremsg->getUeZ();

        it->second.coord.push_back(received_pos);
    }
}

void NrServer::tdoa_positioning()
{
    std::map<int, ue_position_info_t>::iterator it;

    for (it = ue_position_table.begin(); it != ue_position_table.end(); it++)
    {
        double dt21 = it->second.arrive_time[1] - it->second.arrive_time[0];
        double dt31 = it->second.arrive_time[2] - it->second.arrive_time[0];
        double dt41 = it->second.arrive_time[3] - it->second.arrive_time[0];
        ue_coordinate_t& bs1 = it->second.bs_from[0];
        ue_coordinate_t& bs2 = it->second.bs_from[1];
        ue_coordinate_t& bs3 = it->second.bs_from[2];
        ue_coordinate_t& bs4 = it->second.bs_from[3];
        ue_coordinate_t position;

        double light_speed = SPEED_OF_LIGHT;
        double L = light_speed * dt21;
        double R = light_speed * dt31;
        double U = light_speed * dt41;
        // Use taylor direct method to solve the TDOA non-linear equations system
        double XL = bs2.x - bs1.x;
        double YL = bs2.y - bs1.y;
        double ZL = bs2.z - bs1.z;
        double XR = bs3.x - bs1.x;
        double YR = bs3.y - bs1.y;
        double ZR = bs3.z - bs1.z;
        double XU = bs4.x - bs1.x;
        double YU = bs4.y - bs1.y;
        double ZU = bs4.z - bs1.z;
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

        double x1 = MX*k1+NX+bs1.x;
        double y1 = MY*k1+NY+bs1.y;
        double z1 = MZ*k1+NZ+bs1.z;
        double x2 = MX*k2+NX+bs1.x;
        double y2 = MY*k2+NY+bs1.y;
        double z2 = MZ*k2+NZ+bs1.z;

        if (k2 < 0)
        {
            position.x = x1;
            position.y = y1;
            position.z = z1;
        } else {
            double r_ref  = sqrt((x1-bs1.x)*(x1-bs1.x)+(y1-bs1.y)*(y1-bs1.y)+(z1-bs1.z)*(z1-bs1.z));
            double r2_ref = sqrt((x1-bs2.x)*(x1-bs2.x)+(y1-bs2.y)*(y1-bs2.y)+(z1-bs2.z)*(z1-bs2.z));
            double r3_ref = sqrt((x1-bs3.x)*(x1-bs3.x)+(y1-bs3.y)*(y1-bs3.y)+(z1-bs3.z)*(z1-bs3.z));
            double r4_ref = sqrt((x1-bs4.x)*(x1-bs4.x)+(y1-bs4.y)*(y1-bs4.y)+(z1-bs4.z)*(z1-bs4.z));

            if (fabs(k1-k2) < 1.0)
            {
                double r_ref2  = sqrt((x2-bs1.x)*(x2-bs1.x)+(y2-bs1.y)*(y2-bs1.y)+(z2-bs1.z)*(z2-bs1.z));
                double r2_ref2 = sqrt((x2-bs2.x)*(x2-bs2.x)+(y2-bs2.y)*(y2-bs2.y)+(z2-bs2.z)*(z2-bs2.z));
                double r3_ref2 = sqrt((x2-bs3.x)*(x2-bs3.x)+(y2-bs3.y)*(y2-bs3.y)+(z2-bs3.z)*(z2-bs3.z));
                double r4_ref2 = sqrt((x2-bs4.x)*(x2-bs4.x)+(y2-bs4.y)*(y2-bs4.y)+(z2-bs4.z)*(z2-bs4.z));
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
                    position.x = x1;
                    position.y = y1;
                    position.z = z1;
                } else {
                    position.x = x2;
                    position.y = y2;
                    position.z = z2;
                }
            } else {
                if ((fabs((r2_ref - r_ref) - L) < 1E-4) && (fabs((r3_ref - r_ref) - R) < 1E-4) && (abs((r4_ref - r_ref) - U) < 1E-4) && (x1 >= 0) && (
                   y1 >= 0) && (z1>=0)) {
                    position.x = x1;
                    position.y = y1;
                    position.z = z1;
                } else {
                    position.x = x2;
                    position.y = y2;
                    position.z = z2;
                }
            }
        }

        it->second.coord_est.push_back(position);
    } // End of for(:)
}

void NrServer::finish()
{
    std::map<int, ue_position_info_t>::iterator it;

    for (it = ue_position_table.begin(); it != ue_position_table.end(); it++)
    {
        EV << "UE " << it->second.sim_id << " : " << " sizeof(UE_real_positions) = " << it->second.coord.size()
                << " sizeof(UE_est_positions) = " << it->second.coord_est.size() << "\n";
        EV << "  REF: (" << it->second.coord[0].x << " " << it->second.coord[0].y << " " << it->second.coord[0].z << ") \n";
        EV << "  BS1:" << it->second.bs_from[0].x << " " << it->second.bs_from[0].y << " " << it->second.bs_from[0].z << "\n";
        EV << "  BS2:" << it->second.bs_from[1].x << " " << it->second.bs_from[1].y << " " << it->second.bs_from[1].z << "\n";
        EV << "  BS3:" << it->second.bs_from[2].x << " " << it->second.bs_from[2].y << " " << it->second.bs_from[2].z << "\n";
        EV << "  BS4" << it->second.bs_from[3].x << " " << it->second.bs_from[3].y << " " << it->second.bs_from[3].z << "\n";
        EV << "  Arrive time : " << it->second.arrive_time[0] << " " << it->second.arrive_time[1] << " "
                << it->second.arrive_time[2] << " " << it->second.arrive_time[3] << "\n";
        if (it->second.coord_est.size() != 0)
        {
            EV << "  Estimate: (" << it->second.coord_est[0].x << " " << it->second.coord_est[0].y << " " << it->second.coord_est[0].z << ") \n";
        }
    }
}

};

