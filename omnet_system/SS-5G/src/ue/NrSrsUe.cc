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

#include "NrSrsUe.h"

using namespace omnetpp;

namespace ss5G {

Define_Module(NrSrsUe);

void NrSrsUe::initialize()
{
    bs_connected = gateSize("out");

    NrUeBase::initialize();

    char msgname[32];
    sprintf(msgname, "srs-peirod_ind");
    cMessage *msg = new cMessage(msgname);
    srs_control_event = msg;

    scheduleAt(5.0, msg);
    EV << "UE prepare to do SRS period indication, next 5 seconds !\n";
}

void NrSrsUe::handleMessage(cMessage *msg)
{
    if (msg->isSelfMessage()) {
        if (check_state()) {
            srsPeriodForward();
            update_positions();

            EV << "UE do SRS period indication, next 100 ms !\n";
            scheduleAt(simTime() + 0.01, msg);
        } else {
            EV << "UE can not do SRS period indication, wait 5 seconds !\n";
            scheduleAt(simTime() + 5.0, msg);
        }
    } else {
        NrUeBase::handleMessage(msg);
    }
}

void NrSrsUe::update_positions()
{
    if (bs_connected < 4) {
        return;
    }

    for (int i = 0; i < 4; i++) {
        ue2bs_dist[i] = sqrt((x_coord-bs_x_coord[i])*(x_coord-bs_x_coord[i])+(y_coord-bs_y_coord[i])*(y_coord-bs_y_coord[i])+
                (z_coord-bs_z_coord[i])*(z_coord-bs_z_coord[i]));
    }
}

void NrSrsUe::srsPeriodForward()
{
    int src = NrUeBase::sim_id;

    char msgname[64];
    for (int i = 0; i < bs_connected; i++) {
        // Create message object and set source and destination field.
        sprintf(msgname, "ue-%d-srs-ind", src);
        AirFrameMsg *msg = new AirFrameMsg(msgname);
        msg->setSource(src);
        msg->setDestination(-1);
        // Set the message type
        msg->setType(UE_SRS_SIGNAL);
        // Set the coordinates of the UE and the BS destination
        msg->setX(x_coord);
        msg->setY(y_coord);
        msg->setZ(z_coord);
        msg->setDestX(bs_x_coord[i]);
        msg->setDestY(bs_y_coord[i]);
        msg->setDestZ(bs_z_coord[i]);
        // Set the power of the SRS period signal, fixed 30 dB
        msg->setTxPowerUpdate(30.0);
        // Set the time information
        msg->setTimeStamp(0.0);
        EV << "Broadcasting message " << msg << " on BS out[" << i << "]\n";
        send(msg, "out", i);
        numSent++;
    }
}

bool NrSrsUe::check_state()
{
    if (bs_connected <= 0) {
        return false;
    }

    for (int i = 0; i < bs_connected; i++) {
        if (NrUeBase::state[i] != RRC_CONNECTED) {
            return false;
        }
    }

    return true;
}

};

