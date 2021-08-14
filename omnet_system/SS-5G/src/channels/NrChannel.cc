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

#include "NrChannel.h"
#include "../message/airframe_m.h"

using namespace omnetpp;

namespace ss5G {

Define_Module(NrChannel);

void NrChannel::initialize()
{
    distance = par("distance");
    delay = par("delay");

#if 0
    if (hasGUI()) {
        EV << "Disable to show the wireless channel module !\n";
        for (int i = 0; i < gateSize("in"); i++) {
            cGate *gate = this->gate("in", i);
            gate->getDisplayString().setTagArg("ls", 0, "none"); // 1 indicates argument of the Tag, 0 - width
            gate = this->gate("out", i);
            gate->getDisplayString().setTagArg("ls", 0, "none");
        }
    }
#endif
}

void NrChannel::handleMessage(cMessage *msg)
{
    AirFrameMsg *ttmsg = check_and_cast<AirFrameMsg *>(msg);
    int port = ttmsg->getArrivalGate()->getIndex();

    send(msg, "out", port);
}

void NrChannel::setDistance(double dist)
{
    distance = dist;
}

};
