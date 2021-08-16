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

#include "NrPosition.h"
#include "../message/corepacket_m.h"

using namespace omnetpp;

namespace ss5G {

Define_Module(NrPosition);

void NrPosition::initialize()
{
    period_positioning = par("period_positioning");
    if (period_positioning <= 0)
        period_positioning = 0.02;

    position_request_msg = nullptr;
    BsControlMsg *msg = new BsControlMsg("positiong-request");
    msg->setMsgType(POSITION_REQUEST);
    position_request_msg = msg;
    scheduleAt(1E-9, position_request_msg);
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
            EV << "Wait SRS signal to do positioning\n";
        } else {
            AirFrameMsg *ttmsg = check_and_cast<AirFrameMsg *>(msg);

            if (ttmsg->getType() == UE_SRS_SIGNAL) {
                forward2core_positioning_request(ttmsg);
            }
        }

        delete msg;
    }
}


double NrPosition::pathLoss2Distance(int channel_model, double center_freq, double pl)
{
    double dist = 1.0;

    // center_freq is GHz
    if (channel_model == INDOOR_ONLY_PATHLOSS) {
        dist = pow(10, (pl - 32.4 - 20 * log10(center_freq)) / 17.3);
    } else if (channel_model == OUTDOOR_ONLY_PATHLOSS) {
        dist = pow(10, (pl - 28.0 - 20 * log10(center_freq)) / 22);
    } else {
        EV << "Not supported path loss model for the wireless channel in pathLoss2Distance() !\n";
    }

    return dist;
}

void NrPosition::forward2core_positioning_request(AirFrameMsg *ttmsg_ue)
{
    char msgname[64];

    sprintf(msgname, "bs-request-to-core");
    CorePacket *core_msg = new CorePacket(msgname);

    // Set request contents
    int ue_simid = ttmsg_ue->getSource();
    core_msg->setBsSimId(NrGnbBase::sim_id);
    core_msg->setUeSimId(ue_simid);
    core_msg->setBsX(NrGnbBase::x_coord);
    core_msg->setBsY(NrGnbBase::y_coord);
    core_msg->setBsZ(NrGnbBase::z_coord);
    core_msg->setUeX(ttmsg_ue->getX());
    core_msg->setUeY(ttmsg_ue->getY());
    core_msg->setUeZ(ttmsg_ue->getZ());

    // The pass loss, and the SRS signal in tx is always 40 dB
    double pl_power = 40 - ttmsg_ue->getTxPowerUpdate();
    double freq = ttmsg_ue->getCenterFreq();
    int channel_model = ttmsg_ue->getChannelMode();
    double trans_time = pathLoss2Distance(channel_model, freq, pl_power) / SPEED_OF_LIGHT;
    core_msg->setArriveTime(trans_time);
    EV << "After the SRS signal process, the TOA of UE(" << ue_simid << ") is " << trans_time << "\n";

    // Set the time stamp
    double tti_stamp = 0;
    core_msg->setTimeStamp(tti_stamp);

    // Send operation
    core_msg->setType(BS_POSITION_REQ);
    EV << "Send request " << core_msg << " on BS to core network server\n";
    send(core_msg, "core_out");
    NrGnbBase::numSent++;
}

};


