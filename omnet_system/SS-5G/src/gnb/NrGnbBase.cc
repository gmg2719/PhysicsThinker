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

#include "NrGnbBase.h"

using namespace omnetpp;

namespace ss5G {

Define_Module(NrGnbBase);

static int gnb_id_omnetpp = 500;

NrGnbBase::~NrGnbBase()
{
    for (int i = 0; i < ue_backlog_table.size(); i++)
    {
        if (ue_backlog_table[i] != NULL)
        {
            delete ue_backlog_table[i];
        }

        ue_backlog_table[i] = NULL;
    }

    if (control_msg != NULL)
    {
        delete control_msg;
    }
}

void NrGnbBase::initialize()
{
    // Set the ID for the simulation
    sim_id = gnb_id_omnetpp++;
    // Initialize variables
    state = SWITCH_ON_STATE;
    numUeBacklog = gateSize("in");
    numSent = 0;
    numReceived = 0;
    WATCH(numSent);
    WATCH(numReceived);

    // Other variables
    period_sched = par("period_sched");
    if (period_sched <= 0)
        period_sched = 0.0005;
    x_coord = par("x_coord");
    y_coord = par("y_coord");
    z_coord = par("z_coord");

    // Boot the process scheduling the initial message as a self-message.
    control_msg = nullptr;
    BsControlMsg *msg = new BsControlMsg("bcast-info");
    msg->setMsgType(INIT_BROADCAST);
    control_msg = msg;
    scheduleAt(0.0, control_msg);
    EV << "Scheduling first on the Base Station\n";
    EV << "BS schedule period is " << period_sched << " sec !\n";
    EV << "BS coordinate : " << x_coord << " " << y_coord << " " << z_coord << "\n";
}

void NrGnbBase::broadcastPeriodMessage(BsControlMsg *msg)
{
    int src = sim_id;
    int n = gateSize("out");

    char msgname[64];
    for (int i = 0; i < n; i++) {
        // Create message object and set source and destination field.
        sprintf(msgname, "bs-%d-to-%d-port", src, i);
        AirFrameMsg *msg = new AirFrameMsg(msgname);
        msg->setSource(src);
        msg->setDestination(i);
        // Set the message type
        msg->setType(BS_BROAD);
        // Set the coordinates of the BS
        msg->setX(x_coord);
        msg->setY(y_coord);
        msg->setZ(z_coord);
        EV << "Broadcasting message " << msg << " on BS out[" << i << "]\n";
        send(msg, "out", i);
        numSent++;
    }
}

void NrGnbBase::handleMessage(cMessage *msg)
{
    if (state == SWITCH_ON_STATE)
    {
        if (msg->isSelfMessage()) {
            // variable : control_msg
            EV << "Active UEs are " << get_active_ue() << " ...\n";
            if (numUeBacklog == get_active_ue()) {
                if (state == SWITCH_ON_STATE)
                {
                    state = READY_STATE;
                }
            } else {
                BsControlMsg *mst_ct = check_and_cast<BsControlMsg *>(msg);
                if (mst_ct->getMsgType() == INIT_BROADCAST) {
                    broadcastPeriodMessage(mst_ct);
                    // When execute debug, just comment two sentences below, so that broadcast one time and make the handleMessage simpler
                    scheduleAt(simTime() + period_sched, msg);
                    EV << "BS schedule period is next " << period_sched << " sec !\n";
                }
            }
        } else {
            AirFrameMsg *ttmsg = check_and_cast<AirFrameMsg *>(msg);

            EV << "Message from UE arrived, starting to process...\n";
            numReceived++;

            if (ttmsg->getType() == UE_MSG1) {
                EV << "Receive MSG1 from UE " << "sim_id(" << ttmsg->getSource() << ")\n";
                ue_info_t *e = new ue_info_t;
                e->is_active = 0;
                e->in_port_index = ttmsg->getArrivalGate()->getIndex();
                e->sim_id = ttmsg->getSource();
                ue_backlog_table.push_back(e);

                forward_msg2(ttmsg);
                numSent++;
            } else if (ttmsg->getType() == UE_MSG3) {
                forward_msg4(ttmsg);
                numSent++;
            } else if (ttmsg->getType() == UE_COMPLETE_RRC) {
                // Create the RRC connection done.
                ue_info_t *e = find_ue(ttmsg->getSource());
                if (e != NULL) {
                    e->is_active = 1;
                } else {
                    EV << "Simulate error... RRC Setup complete not correct !\n";
                }
            } else {
                EV << "Not supported message !\n";
            }

            delete msg;
        }
    }
}

void NrGnbBase::forward_msg2(AirFrameMsg *ttmsg_ue)
{
    char msgname[64];
    int port = ttmsg_ue->getArrivalGate()->getIndex();

    sprintf(msgname, "bs-msg2-to-UE-%d", ttmsg_ue->getSource());
    AirFrameMsg *ttmsg = new AirFrameMsg(msgname);

    // Set msg2 contents

    // Send operation
    ttmsg->setType(BS_MSG2);
    EV << "Send msg2 " << ttmsg << " on BS out[" << port << "]\n";
    send(ttmsg, "out", port);
}

void NrGnbBase::forward_msg4(AirFrameMsg *ttmsg_ue)
{
    char msgname[64];
    int port = ttmsg_ue->getArrivalGate()->getIndex();

    sprintf(msgname, "bs-msg4-to-UE-%d", ttmsg_ue->getSource());
    AirFrameMsg *ttmsg = new AirFrameMsg(msgname);

    // Set msg4 contents

    // Send operation
    ttmsg->setType(BS_MSG4);
    EV << "Send msg4 " << ttmsg << " on BS out[" << port << "]\n";
    send(ttmsg, "out", port);
}

ue_info_t* NrGnbBase::find_ue(int sim_id)
{
    for (int i = 0; i < ue_backlog_table.size(); i++) {
        if (ue_backlog_table[i]->sim_id == sim_id) {
            return ue_backlog_table[i];
        }
    }

    return NULL;
}

int NrGnbBase::get_active_ue()
{
    int ue_num = 0;

    for (int i = 0; i < ue_backlog_table.size(); i++) {
        if (ue_backlog_table[i]->is_active != 0) {
            ue_num++;
        }
    }

    return ue_num;
}

void NrGnbBase::finish()
{
    // This function is called by OMNeT++ at the end of the simulation.
    EV << "Sent:     " << numSent << endl;
    EV << "Received: " << numReceived << endl;
}

}; //namespace

