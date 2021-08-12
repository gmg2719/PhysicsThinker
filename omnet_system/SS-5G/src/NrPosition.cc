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
#include "corepacket_m.h"

using namespace omnetpp;

namespace ss5G {

Define_Module(NrPosition);

void NrPosition::initialize()
{
    if (period_positioning <= 0) {
        period_positioning = 0.02;
    }

    position_request_msg = nullptr;
    BsControlMsg *msg = new BsControlMsg("positiong-request");
    msg->setMsgType(POSITION_REQUEST);
    position_request_msg = msg;
    scheduleAt(0.0, position_request_msg);
    EV << "Scheduling position request on the Base Station\n";

    NrGnbBase::initialize();
}

void NrPosition::handleMessage(cMessage *msg)
{
    if (NrGnbBase::state < READY_STATE)
    {
        if (msg == position_request_msg) {
            // Period_positioning is 20ms = 20000us
            scheduleAt(simTime() + period_positioning, msg);
            EV << "BS positioning period is next " << period_positioning << " sec !\n";
        } else {
            NrGnbBase::handleMessage(msg);
        }
    }
    else
    {
        if (msg == position_request_msg) {
            forward2core_positioning_request();
            NrGnbBase::numSent++;
        }

        delete msg;
    }
}

void NrPosition::forward2core_positioning_request()
{
    char msgname[64];

    sprintf(msgname, "bs-request-to-core");
    CorePacket *core_msg = new CorePacket(msgname);

    // Set request contents

    // Send operation
    core_msg->setType(BS_POSITION_REQ);
    EV << "Send request " << core_msg << " on BS to core network server\n";
    send(core_msg, "core_out");
}

};


