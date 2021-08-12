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

#include <stdio.h>
#include <string.h>
#include <omnetpp.h>
#include "NrEntity.h"
#include "airframe_m.h"
#include "bscontrol_m.h"

using namespace omnetpp;

namespace ss5G {

Define_Module(NrGnbBase);

void NrGnbBase::initialize()
{
    // Initialize variables
    numSent = 0;
    numReceived = 0;
    WATCH(numSent);
    WATCH(numReceived);

    // Other variables
    if (period_sched <= 0) {
        period_sched = 0.0005;
    }

    // Boot the process scheduling the initial message as a self-message.
    control_msg = nullptr;
    control_msg = new BsControlMsg("bcast-info");
    scheduleAt(0.0, control_msg);
    EV << "Scheduling first on the Base Station\n";
}

void NrGnbBase::broadcastPeriodMessage(BsControlMsg *msg)
{
    int src = -1;
    int n = gateSize("out");

    char msgname[20];
    for (int i = 0; i < n; i++) {
        // Create message object and set source and destination field.
        sprintf(msgname, "bs-to-%d", i);
        AirFrameMsg *msg = new AirFrameMsg(msgname);
        msg->setSource(src);
        msg->setDestination(i);
        EV << "Broadcasting message " << msg << " on BS out[" << i << "]\n";
        send(msg, "out", i);
        numSent++;
    }
}

void NrGnbBase::handleMessage(cMessage *msg)
{
    if (msg == control_msg) {
        BsControlMsg *mst_ct = check_and_cast<BsControlMsg *>(msg);
        broadcastPeriodMessage(mst_ct);

        scheduleAt(simTime() + period_sched, msg);
        EV << "BS schedule period is next " << period_sched << " sec !\n";
    } else {
        EV << "Message from UE arrived, starting to process...\n";
    }
}

void NrGnbBase::finish()
{
    // This function is called by OMNeT++ at the end of the simulation.
    EV << "Sent:     " << numSent << endl;
    EV << "Received: " << numReceived << endl;
}

}; //namespace

