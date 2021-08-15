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

#include <cmath>
#include "NrChannel.h"
#include "../message/airframe_m.h"
#include "../NrEntity.h"

using namespace omnetpp;

namespace ss5G {

Define_Channel(NrChannel);

void NrChannel::initialize()
{
    cDelayChannel::initialize();

    distance = par("distance");
    // Wireless propagate distance in default
    if (distance <= 0) {
        distance = 1.0;
    }
}

void NrChannel::setDistance(double dist)
{
    distance = dist;
}


double NrChannel::path_loss(double dist)
{
    return 31.7 * log10(dist);
}

void NrChannel::processMessage(cMessage *msg, simtime_t t, result_t& result)
{
    AirFrameMsg *ttmsg = check_and_cast<AirFrameMsg *>(msg);

    if (ttmsg->getType() == UE_SRS_SIGNAL)
    {
        double tx_power = ttmsg->getTxPowerUpdate();
        double x = ttmsg->getX();
        double y = ttmsg->getY();
        double z = ttmsg->getZ();
        double dest_x = ttmsg->getDestX();
        double dest_y = ttmsg->getDestY();
        double dest_z = ttmsg->getDestZ();
        double dist = sqrt((x-dest_x)*(x-dest_x)+(y-dest_y)*(y-dest_y)+(z-dest_z)*(z-dest_z));

        double srs_delay = dist / SPEED_OF_LIGHT;

        ttmsg->setTxPowerUpdate(tx_power - path_loss(dist));

        cDelayChannel::setDelay(srs_delay);
        EV << "NrChannel makes propagate delay = " << srs_delay << "\n";
    }
    else
    {
        double normal_delay = distance / SPEED_OF_LIGHT;

        cDelayChannel::setDelay(normal_delay);
    }

    cDelayChannel::processMessage(msg, t, result);
}

};
