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

#ifndef GNB_NRGNBBASE_H_
#define GNB_NRGNBBASE_H_

#include <stdio.h>
#include <string.h>
#include <omnetpp.h>
#include <vector>
#include "../message/airframe_m.h"
#include "../message/bscontrol_m.h"
#include "../NrEntity.h"

namespace ss5G {

using namespace omnetpp;

typedef struct ue_info {
    int is_active;
    int sim_id;
    int in_port_index;
    double propagate_time;
} ue_info_t;

class NrGnbBase : public cSimpleModule
{
  protected:
    int sim_id;
    int state;
    long numSent;
    long numReceived;
    double x_coord;
    double y_coord;
    double z_coord;
    double period_sched;
    cMessage *control_msg;
    int numUeBacklog;
    std::vector<ue_info_t*> ue_backlog_table;

  protected:
    virtual ~NrGnbBase();
    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;

    // Self scheduling
    virtual void broadcastPeriodMessage(BsControlMsg *msg);

    // Initial access
    virtual void forward_msg2(AirFrameMsg *ttmsg_ue);
    virtual void forward_msg4(AirFrameMsg *ttmsg_ue);

    ue_info_t* find_ue(int sim_id);
    int get_active_ue();
    int get_state() { return state; }

    // The finish() function is called by OMNeT++ at the end of the simulation:
    virtual void finish() override;
};

};

#endif /* GNB_NRGNBBASE_H_ */
