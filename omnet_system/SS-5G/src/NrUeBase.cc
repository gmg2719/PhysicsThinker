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

#include "NrEntity.h"
#include "airframe_m.h"

using namespace omnetpp;

namespace ss5G {

Define_Module(NrUeBase);

static int ue_id_omnetpp = 1062;

void NrUeBase::initialize()
{
    // Set the ID for the simulation
    sim_id = ue_id_omnetpp++;
    // Initialize variables
    state = RRC_IDLE;
    numSent = 0;
    numReceived = 0;
    WATCH(numSent);
    WATCH(numReceived);
}

void NrUeBase::handleMessage(cMessage *msg)
{
    AirFrameMsg *ttmsg = check_and_cast<AirFrameMsg *>(msg);
    // Message arrived
    EV << "Message " << ttmsg << " arrived.\n";
    bubble("ARRIVED, bs send something !");

    if (ttmsg->getType() == BS_BROAD) {
        if (state == RRC_IDLE) {
            bs_x_coord = ttmsg->getX();
            bs_y_coord = ttmsg->getY();
            bs_z_coord = ttmsg->getZ();

            forward2Bs_msg1();
            state = RRC_SETUP;
        }
    } else if (ttmsg->getType() == BS_MSG2) {
        forward2Bs_msg3();

    } else if (ttmsg->getType() == BS_MSG4) {
        forward2Bs_msg5();
        state = RRC_CONNECTED;
    } else {
        EV << "UE " << "sim_id(" << sim_id << ")" << " not supported message !\n";
    }

    delete msg;

    // update statistics.
    numReceived++;
}

void NrUeBase::forward2Bs_msg1()
{
    char msgname[64];
    sprintf(msgname, "ue-%d-to-bs-msg1", sim_id);
    AirFrameMsg *msg = new AirFrameMsg(msgname);
    msg->setSource(sim_id);
    msg->setDestination(-1);
    // Set the message type
    msg->setType(UE_MSG1);

    // Set the contents

    EV << "UE send msg1 to the base station.\n";
    send(msg, "out");
    numSent++;
}

void NrUeBase::forward2Bs_msg3()
{
    char msgname[64];
    sprintf(msgname, "ue-%d-to-bs-msg3", sim_id);
    AirFrameMsg *msg = new AirFrameMsg(msgname);
    msg->setSource(sim_id);
    msg->setDestination(-1);
    // Set the message type
    msg->setType(UE_MSG3);

    // Set the contents

    EV << "UE send msg3 to the base station.\n";
    send(msg, "out");
    numSent++;
}

void NrUeBase::forward2Bs_msg5()
{
    char msgname[64];
    sprintf(msgname, "ue-%d-to-bs-msg5", sim_id);
    AirFrameMsg *msg = new AirFrameMsg(msgname);
    msg->setSource(sim_id);
    msg->setDestination(-1);
    // Set the message type
    msg->setType(UE_COMPLETE_RRC);

    // Set the contents

    EV << "UE send RRCSetupComplete msg5 to the base station.\n";
    send(msg, "out");
    numSent++;
}

void NrUeBase::finish()
{
    // This function is called by OMNeT++ at the end of the simulation.
    EV << "Sent:     " << numSent << endl;
    EV << "Received: " << numReceived << endl;
}

}; //namespace

